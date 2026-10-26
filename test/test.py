# SPDX-License-Identifier: Apache-2.0
# Cocotb tests for the 4x4 BF16 systolic GEMM (RTL and gate-level).

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

import bf16_model as M
from gemm_protocol import recv_byte, send_byte

N = K = 4
CHIP_ID = 0xB4

OP_LOAD_A, OP_LOAD_B = 0x10, 0x11
OP_GEMM, OP_CLEAR_GEMM = 0x20, 0x21
OP_READ_C, OP_READ_C_ELEMENT = 0x30, 0x31
OP_STATUS, OP_ID = 0x40, 0x41

ST_BUSY, ST_DONE, ST_A, ST_B, ST_VALID, ST_ERR = (1 << i for i in range(6))


async def reset(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def load(dut, opcode, values):
    await send_byte(dut, opcode)
    for v in values:
        await send_byte(dut, v & 0xFF)
        await send_byte(dut, (v >> 8) & 0xFF)


async def status(dut):
    await send_byte(dut, OP_STATUS)
    return await recv_byte(dut)


async def wait_idle(dut, max_polls=50):
    for _ in range(max_polls):
        s = await status(dut)
        if not s & ST_BUSY:
            return s
    raise AssertionError("busy stuck high")


async def read_c(dut):
    await send_byte(dut, OP_READ_C)
    b = [await recv_byte(dut) for _ in range(2 * N * N)]
    return [b[2 * i] | (b[2 * i + 1] << 8) for i in range(N * N)]


async def run(dut, a, b, accumulate=False):
    await load(dut, OP_LOAD_A, a)
    await load(dut, OP_LOAD_B, b)
    await send_byte(dut, OP_GEMM if accumulate else OP_CLEAR_GEMM)
    return await wait_idle(dut)


def rand_mat(lo=-8.0, hi=8.0):
    return [M.from_float(random.uniform(lo, hi)) for _ in range(N * K)]


@cocotb.test()
async def test_id(dut):
    await reset(dut)
    await send_byte(dut, OP_ID)
    assert await recv_byte(dut) == CHIP_ID


@cocotb.test()
async def test_identity(dut):
    await reset(dut)
    one = M.from_float(1.0)
    ident = [one if i // N == i % N else 0 for i in range(N * K)]
    b = [M.from_float(x) for x in
         [1.5, -2, 3.25, 0.5, 4, 5, -6, 7, 0.125, 8, 9, -10, 11, 12, 13, 14]]
    s = await run(dut, ident, b)
    assert s & ST_DONE and s & ST_A and s & ST_B and s & ST_VALID, f"status {s:#x}"
    assert await read_c(dut) == M.matmul_ref(ident, b, N, K, N)


@cocotb.test()
async def test_random(dut):
    await reset(dut)
    random.seed(1)
    for _ in range(20):
        a, b = rand_mat(), rand_mat()
        await run(dut, a, b)
        assert await read_c(dut) == M.matmul_ref(a, b, N, K, N)


@cocotb.test()
async def test_accumulate(dut):
    await reset(dut)
    random.seed(2)
    a1, b1, a2, b2 = rand_mat(), rand_mat(), rand_mat(), rand_mat()
    await run(dut, a1, b1)
    await run(dut, a2, b2, accumulate=True)
    # C = A1*B1 + A2*B2, equivalent to one K=8 product [A1|A2] x [B1;B2]
    a = [ (a1 if t < K else a2)[i * K + t % K] for i in range(N) for t in range(2 * K)]
    b = b1 + b2
    assert await read_c(dut) == M.matmul_ref(a, b, N, 2 * K, N)


@cocotb.test()
async def test_rerun_without_reload(dut):
    """The prep unit's lanes return to order after a feed, so GEMM can rerun."""
    await reset(dut)
    random.seed(4)
    a, b = rand_mat(), rand_mat()
    await run(dut, a, b)
    first = await read_c(dut)
    await send_byte(dut, OP_CLEAR_GEMM)
    await wait_idle(dut)
    assert await read_c(dut) == first, "clear rerun changed the result"
    await send_byte(dut, OP_GEMM)
    await wait_idle(dut)
    a2 = [a[i * K + t % K] for i in range(N) for t in range(2 * K)]
    assert await read_c(dut) == M.matmul_ref(a2, b + b, N, 2 * K, N)


@cocotb.test()
async def test_read_element(dut):
    await reset(dut)
    random.seed(3)
    a, b = rand_mat(), rand_mat()
    await run(dut, a, b)
    exp = M.matmul_ref(a, b, N, K, N)
    for idx in (0, 5, 10, 15):
        await send_byte(dut, OP_READ_C_ELEMENT)
        await send_byte(dut, idx)
        lo = await recv_byte(dut)
        hi = await recv_byte(dut)
        assert lo | (hi << 8) == exp[idx], f"element {idx}"


@cocotb.test()
async def test_unknown_opcode_sets_error(dut):
    await reset(dut)
    await send_byte(dut, 0x7F)
    assert await status(dut) & ST_ERR
    assert not await status(dut) & ST_ERR, "error should clear after STATUS read"
