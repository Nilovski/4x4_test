![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg)

# 4x4 BF16 output-stationary systolic array

A 4x4 output-stationary **BF16** systolic matrix-multiply accelerator for
Tiny Tapeout (sky130): `C = A x B` (CLEAR_GEMM) or `C += A x B` (GEMM), with
real BF16 multiplies and BF16 accumulation, driven over an 8-bit byte-serial
host protocol. The host tiles matrix products of any size onto the array.

- Top module: `tt_um_apex_bf16_systolic`
- Tiles: 6x2
- Chip ID: `0xA8`
- Protocol, pinout and test procedure: [docs/info.md](docs/info.md)

## Architecture

```
host bytes -> rx_byte_interface -> command_decoder -> data_prep_unit (A/B lanes, skewed feed)
                                         |                    |
                                  gemm_controller --beat--> systolic_array
                                         |                 16 x systolic_pe (A_REG, B_REG, ACC)
                                  status_register           4 x pe_row_engine -> bf16_mac_seq
                                         |                    |
host bytes <- tx_byte_interface <- command_decoder <- result_mux
```

- A moves east and B moves south one PE per beat; C stays in each PE.
- Each row shares one multi-cycle BF16 MAC (`bf16_mac_seq`, at most 27
  cycles) that updates the row's 4 accumulators every 128-cycle beat.
- Arithmetic is bit-exact with `test/bf16_model.py` (RNE, flush-to-zero,
  canonical NaN). `src/compute/bf16_multiplier.sv` and `bf16_adder.sv` are
  the original combinational reference units (not in the build).
- An 8x8 BF16 array does not fit in the largest Tiny Tapeout size (8x2): the
  operand buffers and 64 PEs alone exceed it, so the array is 4x4 and larger
  products are tiled by the host.

## Repository layout

```
src/        RTL (interface/, prep/, compute/, control/, output/)
test/       cocotb full-chip tests (test.py), host protocol + tiling (gemm_protocol.py),
            BF16 reference model (bf16_model.py)
test/mac/   standalone test of bf16_mac_seq against the model
host/       demo-board runner (MicroPython + microcotb)
docs/       datasheet (info.md)
```

## Running the tests

```sh
pip install -r test/requirements.txt
cd test && make -B           # full chip
cd mac && make -B            # BF16 MAC unit, ~28k vectors incl. NaN/inf/subnormals
```

The full-chip suite covers ID and pins, identity, A x I = A, I x B = B, all
ones, random products, accumulate, rerun without reload, element reads,
NaN/inf/overflow/subnormal handling, cancellation, STATUS bits, LOAD/GEMM
while busy, `ena` low, a slow host on READ_C, and host-tiled 8x8 and
5x9 @ 9x6 products. Gate-level simulation runs in CI after hardening.

## Running on the demo board

```python
import tt_um_apex_bf16_systolic as g
g.main()
g.matmul([[1.5, -2.0], [0.25, 3.0]], [[2.0, 1.0], [4.0, -0.5]])
```

## License

Apache 2.0. See [LICENSE](LICENSE).
