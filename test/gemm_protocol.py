# SPDX-License-Identifier: Apache-2.0
#
# Host side of the tt_um_apex_bf16_systolic byte protocol.
#
# Shared by the cocotb RTL/gate-level tests (test/test.py) and the Tiny Tapeout
# demo board runner (host/tt_um_apex_bf16_systolic.py, via microcotb), so the
# code that passes in simulation is the code that drives the chip.
#
# All coroutines take a cocotb-style `dut` exposing ui_in, uo_out, uio_in,
# uio_out and clk. Matrices are row-major lists of BF16 bit patterns (ints);
# use bf16_model.from_float / to_float to convert.

try:
    from cocotb.triggers import RisingEdge
except ImportError:  # TT demo board: microcotb provides the same API
    from microcotb.triggers import RisingEdge

import bf16_model as M

N = 4  # array size (C block is N x N)
K = 4  # inner dimension per run

# uio bit positions
RX_VALID = 0
RX_READY = 1
TX_READY = 2
TX_VALID = 3
BUSY = 4
DONE = 5

# the RP2/host drives these uio bits (everything else is an input to the host)
HOST_UIO_OE = (1 << RX_VALID) | (1 << TX_READY)

OP_NOP = 0x00
OP_LOAD_A = 0x10  # + 2*N*K bytes, BF16 row-major, low byte first
OP_LOAD_B = 0x11
OP_GEMM = 0x20  # C += A x B
OP_CLEAR_GEMM = 0x21  # C  = A x B
OP_READ_C = 0x30  # replies 2*N*N bytes, C row-major, low byte first
OP_READ_C_ELEMENT = 0x31  # + index (row*N + col); replies 2 bytes
OP_STATUS = 0x40
OP_ID = 0x41
CHIP_ID = 0xA8

# STATUS bits
ST_BUSY = 0x01
ST_DONE = 0x02
ST_A_LOADED = 0x04
ST_B_LOADED = 0x08
ST_RESULT_VALID = 0x10
ST_ERROR = 0x20


class GemmTimeout(Exception):
    """Chip did not answer (MicroPython has no built-in TimeoutError)."""


# ---------------------------------------------------------------------------
# byte-level protocol
# ---------------------------------------------------------------------------
_host_uio = {}  # uio bits driven by the host, kept here rather than read back


def _set_uio_bit(dut, bit, val):
    # Track the value locally: reading uio_in back can return a stale value
    # when a previous write has not been applied yet (cocotb defers writes).
    cur = _host_uio.get(id(dut), 0)
    cur = (cur | (1 << bit)) if val else (cur & ~(1 << bit))
    _host_uio[id(dut)] = cur
    dut.uio_in.value = cur


async def send_byte(dut, value):
    """One byte on ui_in with a one-cycle RX_VALID pulse (RX_READY is tied high)."""
    dut.ui_in.value = value & 0xFF
    _set_uio_bit(dut, RX_VALID, 1)
    await RisingEdge(dut.clk)
    _set_uio_bit(dut, RX_VALID, 0)
    await RisingEdge(dut.clk)


async def recv_byte(dut, timeout_cycles=50):
    """Wait for TX_VALID, capture uo_out, pulse TX_READY to accept it."""
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if int(dut.uio_out.value) & (1 << TX_VALID):
            byte = int(dut.uo_out.value)
            _set_uio_bit(dut, TX_READY, 1)
            await RisingEdge(dut.clk)
            _set_uio_bit(dut, TX_READY, 0)
            return byte
    raise GemmTimeout("timed out waiting for TX_VALID")


async def load_matrix(dut, opcode, values):
    """LOAD_A / LOAD_B with N*K BF16 values (bit patterns)."""
    await send_byte(dut, opcode)
    for v in values:
        await send_byte(dut, v & 0xFF)
        await send_byte(dut, (v >> 8) & 0xFF)


async def read_id(dut):
    await send_byte(dut, OP_ID)
    return await recv_byte(dut)


async def status(dut):
    """Raw STATUS byte: busy, done, A_loaded, B_loaded, result_valid, error."""
    await send_byte(dut, OP_STATUS)
    return await recv_byte(dut)


async def wait_idle(dut, max_polls=2000):
    """Poll STATUS until busy drops; returns the final STATUS byte."""
    for _ in range(max_polls):
        s = await status(dut)
        if not s & ST_BUSY:
            return s
    raise GemmTimeout("GEMM never finished (busy stuck high)")


async def read_c(dut):
    """READ_C: the N*N result block as BF16 bit patterns."""
    await send_byte(dut, OP_READ_C)
    raw = [await recv_byte(dut) for _ in range(2 * N * N)]
    return [raw[2 * i] | (raw[2 * i + 1] << 8) for i in range(N * N)]


async def read_c_element(dut, index):
    """READ_C_ELEMENT: one element, index = row*N + col."""
    await send_byte(dut, OP_READ_C_ELEMENT)
    await send_byte(dut, index)
    lo = await recv_byte(dut)
    hi = await recv_byte(dut)
    return lo | (hi << 8)


async def gemm4(dut, a, b, accumulate=False):
    """One 4x4x4 run: load A and B (BF16 bits), start, wait. Returns STATUS."""
    await load_matrix(dut, OP_LOAD_A, a)
    await load_matrix(dut, OP_LOAD_B, b)
    await send_byte(dut, OP_GEMM if accumulate else OP_CLEAR_GEMM)
    return await wait_idle(dut)


# ---------------------------------------------------------------------------
# arbitrary-size matmul by tiling onto the 4x4 array
# ---------------------------------------------------------------------------
def _block(mat, rows, cols, r0, c0):
    """4x4 block at (r0, c0) of a rows x cols row-major matrix, zero-padded."""
    out = []
    for r in range(r0, r0 + N):
        for c in range(c0, c0 + K):
            out.append(mat[r * cols + c] if (r < rows and c < cols) else 0)
    return out


async def matmul(dut, a, b, n, k, m):
    """C = A x B for any n x k by k x m BF16 matrices (row-major bit lists).

    Each 4x4 output block is computed on-chip: CLEAR_GEMM for the first
    K-block, then GEMM (accumulate) for the rest, so every element is one
    BF16 multiply-accumulate chain, exactly as bf16_model.matmul_ref.
    """
    c = [0] * (n * m)
    for bi in range(0, n, N):
        for bj in range(0, m, N):
            for idx, kb in enumerate(range(0, k, K)):
                await gemm4(
                    dut,
                    _block(a, n, k, bi, kb),
                    _block(b, k, m, kb, bj),
                    accumulate=(idx > 0),
                )
            part = await read_c(dut)
            for r in range(N):
                for col in range(N):
                    if bi + r < n and bj + col < m:
                        c[(bi + r) * m + (bj + col)] = part[r * N + col]
    return c


def matmul_ref(a, b, n, k, m):
    """Bit-exact reference for matmul()."""
    return M.matmul_ref(a, b, n, k, m, kpad=K)
