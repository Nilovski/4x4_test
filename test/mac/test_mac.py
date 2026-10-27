# SPDX-License-Identifier: Apache-2.0
# bf16_mac_seq vs the bit-exact model: result == add(acc, mul(a, b)).
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

import bf16_model as m

SPECIAL = [0x0000, 0x8000, 0x0001, 0x807F, 0x3F80, 0xBF80, 0x7F80, 0xFF80,
           0x7FC0, 0x7F81, 0x7F7F, 0xFF7F, 0x0080, 0x8080, 0x3FFF, 0x4000,
           0x3F81, 0x00FF, 0x7F00, 0x0100]

worst = [0]


async def run(dut, acc, a, b):
    dut.a.value = a
    dut.b.value = b
    dut.acc.value = acc
    dut.go.value = 1
    await RisingEdge(dut.clk)
    dut.go.value = 0
    for n in range(1, 64):
        await RisingEdge(dut.clk)
        if int(dut.done.value):
            worst[0] = max(worst[0], n)
            return int(dut.result.value)
    raise AssertionError("MAC never finished")


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.go.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


def rand_bf16(rng):
    r = rng.random()
    if r < 0.1:
        return rng.choice(SPECIAL)
    if r < 0.4:   # narrow exponent range: lots of cancellation / ties
        return rng.getrandbits(1) << 15 | rng.randint(120, 134) << 7 | rng.getrandbits(7)
    return rng.getrandbits(16)


@cocotb.test()
async def test_specials(dut):
    await setup(dut)
    for acc in SPECIAL:
        for a in SPECIAL:
            for b in SPECIAL[::3]:
                got = await run(dut, acc, a, b)
                exp = m.mac(acc, a, b)
                assert got == exp, f"acc={acc:04x} a={a:04x} b={b:04x}: got {got:04x} exp {exp:04x}"


@cocotb.test()
async def test_random(dut):
    await setup(dut)
    rng = random.Random(1)
    for i in range(20000):
        acc, a, b = rand_bf16(rng), rand_bf16(rng), rand_bf16(rng)
        got = await run(dut, acc, a, b)
        exp = m.mac(acc, a, b)
        assert got == exp, f"#{i} acc={acc:04x} a={a:04x} b={b:04x}: got {got:04x} exp {exp:04x}"
    dut._log.info(f"worst-case latency: {worst[0]} cycles")
