# SPDX-License-Identifier: Apache-2.0
#
# Hardware-in-the-loop runner for tt_um_apex_bf16_systolic on the Tiny Tapeout
# demo board (RP2040/RP2350 + tt-micropython-firmware, which ships microcotb).
#
# It reuses ../test/gemm_protocol.py and ../test/bf16_model.py -- the exact
# host code and reference model the cocotb simulation tests use.
#
# Install (from the repo root, board on USB):
#     mpremote cp test/bf16_model.py :
#     mpremote cp test/gemm_protocol.py :
#     mpremote cp host/tt_um_apex_bf16_systolic.py :
# Run on the board's REPL:
#     >>> import tt_um_apex_bf16_systolic as g
#     >>> g.main()                                    # self-test on silicon
#     >>> g.matmul([[1.5, -2.0]], [[2.0], [4.0]])     # your own matrices
#
# Pins: the RP2 drives ui_in[7:0], uio[0] RX_VALID and uio[2] TX_READY; it
# reads uo_out[7:0], uio[1] RX_READY, uio[3] TX_VALID, uio[4] BUSY, uio[5] DONE.

import random

import microcotb as cocotb
from microcotb.clock import Clock
from microcotb.triggers import ClockCycles

cocotb.set_runner_scope(__name__)

from ttboard.demoboard import DemoBoard  # noqa: E402
from ttboard.mode import RPMode  # noqa: E402

import bf16_model as M  # noqa: E402
import gemm_protocol as gp  # noqa: E402

PROJECT = "tt_um_apex_bf16_systolic"
CLOCK_PERIOD_US = 10  # bit-banged project clock; the protocol is clock-agnostic


async def reset(dut):
    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD_US, units="us").start())
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    gp._host_uio.clear()
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


def _rand_bf16(count, lo=-4.0, hi=4.0):
    return [M.from_float(random.uniform(lo, hi)) for _ in range(count)]


@cocotb.test()
async def test_id(dut):
    await reset(dut)
    got = await gp.read_id(dut)
    assert got == gp.CHIP_ID, "ID {:#x} != {:#x}".format(got, gp.CHIP_ID)


@cocotb.test()
async def test_ones(dut):
    await reset(dut)
    one = M.from_float(1.0)
    await gp.gemm4(dut, [one] * 16, [one] * 16)
    assert await gp.read_c(dut) == [M.from_float(4.0)] * 16


@cocotb.test()
async def test_random_4x4(dut):
    await reset(dut)
    random.seed(1)
    for _ in range(3):
        a, b = _rand_bf16(16), _rand_bf16(16)
        await gp.gemm4(dut, a, b)
        assert await gp.read_c(dut) == gp.matmul_ref(a, b, 4, 4, 4)


@cocotb.test()
async def test_tiled_8x8(dut):
    await reset(dut)
    random.seed(3)
    a, b = _rand_bf16(64), _rand_bf16(64)
    assert await gp.matmul(dut, a, b, 8, 8, 8) == gp.matmul_ref(a, b, 8, 8, 8)


# ---------------------------------------------------------------------------
# user matmul: registered in its own runner scope so main() doesn't run it
USER_SCOPE = PROJECT + "_user"
_job = {}

cocotb.set_runner_scope(USER_SCOPE)


@cocotb.test()
async def user_matmul(dut):
    await reset(dut)
    n, k, m = _job["dims"]
    _job["c"] = await gp.matmul(dut, _job["a"], _job["b"], n, k, m)


cocotb.set_runner_scope(__name__)


# ---------------------------------------------------------------------------
def _setup():
    tt = DemoBoard.get()
    getattr(tt.shuttle, PROJECT).enable()
    if tt.mode != RPMode.ASIC_RP_CONTROL:
        tt.mode = RPMode.ASIC_RP_CONTROL
    tt.uio_oe_pico.value = gp.HOST_UIO_OE  # RX_VALID, TX_READY driven by us
    return tt


def _make_dut():
    import ttboard.cocotb.dut

    class DUT(ttboard.cocotb.dut.DUT):
        def __init__(self):
            super().__init__("bf16_systolic")

    return DUT()


def main():
    """Run the self-tests above against the chip."""
    _setup()
    runner = cocotb.get_runner(__name__)
    runner.test(_make_dut())
    return runner


def matmul(A, B, check=True):
    """C = A x B on the chip for your own matrices (lists of rows of numbers).

    A is n x k, B is k x m, any size. Values are rounded to BF16, every
    product and sum is done in BF16 on the chip, and C is printed and
    returned as a list of rows of floats. With check=True the result is
    compared bit for bit with the Python model of the chip.
    """
    n, k = len(A), len(A[0])
    if len(B) != k:
        raise ValueError("A is {}x{} but B has {} rows".format(n, k, len(B)))
    m = len(B[0])
    a = [M.from_float(float(v)) for row in A for v in row]
    b = [M.from_float(float(v)) for row in B for v in row]
    if len(a) != n * k or len(b) != k * m:
        raise ValueError("rows must all have the same length")
    _setup()
    _job.clear()
    _job["a"], _job["b"], _job["dims"] = a, b, (n, k, m)
    cocotb.get_runner(USER_SCOPE).test(_make_dut())
    c = _job.get("c")
    if c is None:
        print("matmul did not complete")
        return None
    rows = [[M.to_float(v) for v in c[r * m:(r + 1) * m]] for r in range(n)]
    for row in rows:
        print(" ".join("{:12.5g}".format(v) for v in row))
    if check:
        print("matches BF16 model:", c == gp.matmul_ref(a, b, n, k, m))
    return rows
