# SPDX-License-Identifier: Apache-2.0
#
# Bit-exact Python model of the chip's BF16 arithmetic (src/compute/bf16_*.sv).
#
# Format: 1 sign, 8 exponent (bias 127), 7 mantissa bits.
#   * Subnormal inputs are treated as zero and results below the normal range
#     flush to signed zero (FTZ/DAZ).
#   * Round to nearest, ties to even.
#   * NaN in -> canonical quiet NaN 0x7FC0 out. inf*0 and inf-inf -> NaN.
#   * Exact zero from x + (-x) is +0.
#
# Integer-only so it runs unchanged under MicroPython on the demo board.

QNAN = 0x7FC0
POS_INF = 0x7F80


def _fields(x):
    return (x >> 15) & 1, (x >> 7) & 0xFF, x & 0x7F


def is_nan(x):
    return (x >> 7) & 0xFF == 0xFF and x & 0x7F != 0


def is_inf(x):
    return (x >> 7) & 0xFF == 0xFF and x & 0x7F == 0


def is_zero(x):
    """Zero or subnormal (both read as zero by the hardware)."""
    return (x >> 7) & 0xFF == 0


def _pack(s, e, mant):
    """Round-free pack with range handling. mant includes the hidden bit."""
    if e >= 0xFF:
        return (s << 15) | POS_INF
    if e <= 0:
        return s << 15
    return (s << 15) | (e << 7) | (mant & 0x7F)


def _round(mant, g, sticky, e):
    """RNE on an 8-bit significand with guard bit g and sticky bit."""
    if g and (sticky or (mant & 1)):
        mant += 1
        if mant == 0x100:
            mant = 0x80
            e += 1
    return mant, e


def mul(a, b):
    sa, ea, ma = _fields(a)
    sb, eb, mb = _fields(b)
    s = sa ^ sb
    if is_nan(a) or is_nan(b):
        return QNAN
    if is_inf(a) or is_inf(b):
        if is_zero(a) or is_zero(b):
            return QNAN
        return (s << 15) | POS_INF
    if ea == 0 or eb == 0:
        return s << 15
    p = (0x80 | ma) * (0x80 | mb)  # 16 bits, top bit 14 or 15
    e = ea + eb - 127
    if p & 0x8000:
        mant, g, sticky = (p >> 8) & 0xFF, (p >> 7) & 1, (p & 0x7F) != 0
        e += 1
    else:
        mant, g, sticky = (p >> 7) & 0xFF, (p >> 6) & 1, (p & 0x3F) != 0
    mant, e = _round(mant, g, sticky, e)
    return _pack(s, e, mant)


def add(a, b):
    sa, ea, ma = _fields(a)
    sb, eb, mb = _fields(b)
    if is_nan(a) or is_nan(b):
        return QNAN
    if is_inf(a) and is_inf(b):
        return a if sa == sb else QNAN
    if is_inf(a):
        return a
    if is_inf(b):
        return b
    if ea == 0 and eb == 0:
        return (sa & sb) << 15
    if ea == 0:
        return b
    if eb == 0:
        return a
    # order so that |a| >= |b|
    if (ea << 7 | ma) < (eb << 7 | mb):
        sa, ea, ma, sb, eb, mb = sb, eb, mb, sa, ea, ma
    d = ea - eb
    big = (0x80 | ma) << 3  # 11 bits: hidden, 7 mantissa, guard, round, sticky
    small = (0x80 | mb) << 3
    if d >= 11:
        small = 1
    else:
        sticky = 1 if small & ((1 << d) - 1) else 0
        small = (small >> d) | sticky
    e = ea
    if sa == sb:
        r = big + small
        if r & 0x800:
            r = (r >> 1) | (r & 1)
            e += 1
    else:
        r = big - small
        if r == 0:
            return 0
        while not r & 0x400:
            r <<= 1
            e -= 1
    mant, e = _round(r >> 3, (r >> 2) & 1, (r & 3) != 0, e)
    return _pack(sa, e, mant)


def mac(acc, a, b):
    return add(acc, mul(a, b))


# ---------------------------------------------------------------------------
# float conversion (host convenience; not used by the hardware path)
# ---------------------------------------------------------------------------
def from_float(x):
    """Python float -> BF16 bits, RNE, FTZ."""
    import struct

    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    if (bits >> 23) & 0xFF == 0xFF and bits & 0x7FFFFF:
        return QNAN
    lsb = (bits >> 16) & 1
    bits = (bits + 0x7FFF + lsb) & 0xFFFFFFFF
    h = bits >> 16
    return (h & 0x8000) if is_zero(h) else h


def to_float(h):
    import struct

    return struct.unpack("<f", struct.pack("<I", (h & 0xFFFF) << 16))[0]


# ---------------------------------------------------------------------------
# matrix reference, in the same order the array accumulates
# ---------------------------------------------------------------------------
def matmul_ref(a, b, n, k, m, kpad=4):
    """a: n*k, b: k*m row-major BF16 bits -> n*m. The inner dimension is
    zero-padded to a multiple of kpad, exactly as the host tiles it."""
    kp = -(-k // kpad) * kpad
    out = []
    for i in range(n):
        for j in range(m):
            acc = 0
            for t in range(kp):
                av = a[i * k + t] if t < k else 0
                bv = b[t * m + j] if t < k else 0
                acc = mac(acc, av, bv)
            out.append(acc)
    return out
