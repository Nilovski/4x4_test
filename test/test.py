# SPDX-License-Identifier: Apache-2.0
# Cocotb tests for the 4x4 BF16 systolic GEMM (RTL and gate-level).

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

import bf16_model as M
from gemm_protocol import _host_uio, recv_byte, send_byte
import gemm_protocol as gp

N = K = 4
CHIP_ID = 0xA8

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
    _host_uio.clear()
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


async def wait_idle(dut, max_polls=2000):
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


# ---------------------------------------------------------------------------
# plan test cases and edge cases
# ---------------------------------------------------------------------------
def f(xs):
    return [M.from_float(x) for x in xs]


@cocotb.test()
async def test_plan_cases(dut):
    """A x I = A, I x B = B, all ones -> all K."""
    await reset(dut)
    random.seed(5)
    one = M.from_float(1.0)
    ident = [one if i // N == i % N else 0 for i in range(N * K)]
    a = rand_mat()
    await run(dut, a, ident)
    assert await read_c(dut) == a, "A x I != A"
    await run(dut, ident, a)
    assert await read_c(dut) == a, "I x B != B"
    await run(dut, [one] * 16, [one] * 16)
    assert await read_c(dut) == [M.from_float(float(K))] * 16, "ones"


@cocotb.test()
async def test_specials(dut):
    """NaN, inf, inf*0, inf-inf, zeros, subnormals (read as 0), overflow."""
    await reset(dut)
    inf, ninf, nan = 0x7F80, 0xFF80, 0x7FC0
    big = M.from_float(3.0e38)
    tiny = 0x0001                        # subnormal: reads as zero
    a = [inf, 0, 0, 0,   nan, 0, 0, 0,   big, big, 0, 0,   tiny, 0x8000, 0, 0]
    b = [M.from_float(2.0), inf, 0, 0,
         0, ninf, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0]
    b[0] = M.from_float(2.0); b[1] = 0    # row 0 of B
    b[4] = M.from_float(2.0); b[5] = ninf # row 1 of B
    await run(dut, a, b)
    got = await read_c(dut)
    exp = M.matmul_ref(a, b, N, K, N)
    assert got == exp, f"got {[hex(x) for x in got]} exp {[hex(x) for x in exp]}"
    assert M.is_inf(got[0]) and M.is_nan(got[1]), "inf*2 and inf*0 + ... "
    assert M.is_inf(got[2 * N]), "3e38*2 + 3e38*2 should overflow to inf"


@cocotb.test()
async def test_cancellation(dut):
    """x + (-x) = +0 and near-cancellation needs many normalize steps."""
    await reset(dut)
    a = f([1.0, -1.0, 1.0, -1.0] * 4)
    b = f([1.5, 1.5, 1.5, 1.5,
           1.5, 1.5, 1.5, 1.5,
           1.0078125, 1.0, 1.0, 1.0,
           1.0, 1.0078125, 1.0, 1.0])
    await run(dut, a, b)
    assert await read_c(dut) == M.matmul_ref(a, b, N, K, N)


@cocotb.test()
async def test_status_bits_and_busy(dut):
    """STATUS bits per the plan; LOAD while busy is swallowed; GEMM while busy
    is ignored and flags error."""
    await reset(dut)
    assert await status(dut) == 0
    random.seed(6)
    a, b = rand_mat(), rand_mat()
    await load(dut, OP_LOAD_A, a)
    assert await status(dut) == ST_A
    await load(dut, OP_LOAD_B, b)
    assert await status(dut) == ST_A | ST_B
    await send_byte(dut, OP_CLEAR_GEMM)
    await ClockCycles(dut.clk, 2)
    assert int(dut.uio_out.value) & (1 << gp.BUSY), "BUSY pin"
    await load(dut, OP_LOAD_A, [0x3F80] * 16)        # swallowed
    s = await status(dut)
    assert s & ST_BUSY and s & ST_ERR and not s & ST_DONE, f"mid-run {s:#x}"
    s = await wait_idle(dut)
    assert s & ST_DONE and s & ST_VALID, f"{s:#x}"
    assert int(dut.uio_out.value) & (1 << gp.DONE), "DONE pin"
    assert await read_c(dut) == M.matmul_ref(a, b, N, K, N), "LOAD during busy corrupted A"
    await send_byte(dut, OP_CLEAR_GEMM)
    await wait_idle(dut)
    assert await read_c(dut) == M.matmul_ref(a, b, N, K, N), "stored A changed"


@cocotb.test()
async def test_ena_low_ignores_input(dut):
    await reset(dut)
    dut.ena.value = 0
    await send_byte(dut, OP_ID)
    for _ in range(10):
        await ClockCycles(dut.clk, 1)
        assert not int(dut.uio_out.value) & (1 << gp.TX_VALID)
    dut.ena.value = 1
    await send_byte(dut, OP_ID)
    assert await recv_byte(dut) == CHIP_ID
    assert int(dut.uio_oe.value) == 0b0011_1010


@cocotb.test()
async def test_read_c_slow_host(dut):
    """READ_C waits for TX_READY on every byte."""
    await reset(dut)
    random.seed(7)
    a, b = rand_mat(), rand_mat()
    await run(dut, a, b)
    await send_byte(dut, OP_READ_C)
    raw = []
    for _ in range(2 * N * N):
        await ClockCycles(dut.clk, 9)
        raw.append(await recv_byte(dut))
    got = [raw[2 * i] | raw[2 * i + 1] << 8 for i in range(N * N)]
    assert got == M.matmul_ref(a, b, N, K, N)


@cocotb.test()
async def test_tiled_matmul(dut):
    """Host tiling: 8x8 and a ragged 5x9 by 9x6 product, bit-exact."""
    await reset(dut)
    random.seed(8)
    for n, k, m in ((8, 8, 8), (5, 9, 6)):
        a = [M.from_float(random.uniform(-4, 4)) for _ in range(n * k)]
        b = [M.from_float(random.uniform(-4, 4)) for _ in range(k * m)]
        assert await gp.matmul(dut, a, b, n, k, m) == gp.matmul_ref(a, b, n, k, m), (n, k, m)
