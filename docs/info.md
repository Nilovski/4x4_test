## How it works

A 4x4 **output-stationary BF16 systolic array** computes **C = A x B** or
**C += A x B**, where A, B and C are 4x4 matrices of BF16 values (1 sign,
8 exponent, 7 mantissa bits). Every multiply and every add is real BF16
arithmetic, round to nearest even, so each element of C is the BF16
multiply-accumulate chain C[i][j] = (((0 + A[i][0]B[0][j]) + A[i][1]B[1][j]) + ...).

**Dataflow.** A enters from the west edge and moves east one PE per *beat*;
B enters from the north and moves south. Row *i* and column *j* are delayed by
*i* and *j* beats, so PE(i,j) holds A[i][k] and B[k][j] together in beat
k + i + j. C[i][j] stays in PE(i,j). A run is K + 2(N-1) = 10 beats.

**Processing elements.** Each PE has an A register, a B register and a BF16
accumulator. The arithmetic is done by one **multi-cycle BF16 MAC per row**,
shared by that row's 4 PEs: during each beat it computes
ACC = ACC + A_REG x B_REG for every PE whose operands are valid and writes the
result back. The MAC rounds the product to BF16 and then adds it to the
accumulator. It uses an 8-step shift-add multiplier, one-bit-per-cycle
alignment and normalization, and a shared rounder (at most 27 cycles per
MAC). A beat is 128 clock cycles, so a full run takes about 1,300 cycles
(130 us at 10 MHz).

Sharing the MAC keeps the area at about half of a design with a full
combinational BF16 multiplier and adder in every PE, which did not route even
in 8x2 tiles.

**Number format details**

- Subnormal inputs are read as zero; results below the normal range flush to
  signed zero.
- Overflow gives signed infinity.
- Any NaN input, inf x 0 or inf - inf gives the quiet NaN `0x7FC0`.
- x + (-x) = +0.

A **data preparation unit** holds A and B in 4 + 4 lane buffers (one per
array row and column) and feeds them to the array edges with the skew above,
one step per beat. The lanes rotate back to their original order after a run,
so the same operands can be run again without reloading.

Because results can accumulate across runs, a host can compute a matrix
product of **any size** on this 4x4 array: it splits the operands into 4x4
blocks and accumulates the K-blocks of each output block on-chip with GEMM.

### Protocol

Two synchronous handshake channels carry bytes. `RX_VALID` + `ui[7:0]` bring
a byte in (`RX_READY` is always high), and `TX_VALID` + `uo[7:0]` send one
out, held until the host pulses `TX_READY`.

| Byte   | Op             | Followed by                          | Reply                         |
|--------|----------------|--------------------------------------|-------------------------------|
| `0x00` | NOP            | --                                   | none                          |
| `0x10` | LOAD_A         | 32 bytes: 16 BF16, row-major, low byte first | none                  |
| `0x11` | LOAD_B         | 32 bytes: 16 BF16, row-major, low byte first | none                  |
| `0x20` | GEMM           | --                                   | none (C += A x B)             |
| `0x21` | CLEAR_GEMM     | --                                   | none (C = A x B)              |
| `0x30` | READ_C         | --                                   | 32 bytes: C row-major, low byte first |
| `0x31` | READ_C_ELEMENT | 1 index byte, row*4 + col            | 2 bytes: low, high            |
| `0x40` | STATUS         | --                                   | status byte (below)           |
| `0x41` | ID             | --                                   | `0xA8`                        |

STATUS bits: 0 busy, 1 done, 2 A_loaded, 3 B_loaded, 4 result_valid,
5 error, 7:6 reserved.

- **done** is sticky: set when a run finishes, cleared by the next GEMM or
  CLEAR_GEMM. It is also on pin `uio[5]` (usable as an interrupt line).
- **error** is set by an unknown opcode, a LOAD/GEMM/CLEAR_GEMM/READ while
  busy, an out-of-range element index, or a byte received while a reply is
  being sent. Reading STATUS clears it.
- A LOAD sent while busy has its data bytes consumed and discarded, so the
  stored operands stay intact and the byte stream stays in sync.
- **BUSY** (`uio[4]`) is high while a run is in progress.

## How to test

1. Reset, then send `0x41` (ID) and expect `0xA8`.
2. Send `0x10` and then 16 BF16 values of A (row-major, low byte then high
   byte). Send `0x11` and then 16 BF16 values of B.
3. Send `0x21` (CLEAR_GEMM) to start.
4. Wait for `uio[5]` DONE, or poll `0x40` until bit 0 (busy) reads 0.
5. Send `0x30` (READ_C) and collect 32 bytes: C00 low, C00 high, C01 low, ...

To accumulate, load new A/B blocks and send `0x20` instead of `0x21`.

Example: A = all 1.0 (`0x3F80`), B = all 1.0, gives C = all 4.0 (`0x4080`).

**On the Tiny Tapeout demo board**, copy `test/bf16_model.py`,
`test/gemm_protocol.py` and `host/tt_um_apex_bf16_systolic.py` to the RP2 with
`mpremote`. Then, in the REPL:

```python
import tt_um_apex_bf16_systolic as g
g.main()                                          # self-tests on silicon
g.matmul([[1.5, -2.0], [0.25, 3.0]], [[2.0, 1.0], [4.0, -0.5]])   # your own matrices, any size
```

These scripts use the same protocol and tiling code as the cocotb simulation
tests in `test/test.py`.

## External hardware

None. All you need is a host (the demo board's RP2040/RP2350, or any MCU or
FPGA) driving `ui`/`uo`/`uio[5:0]`.
