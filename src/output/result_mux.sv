`default_nettype none

// ---------------------------------------------------------------------------
// result_mux -- selects one byte of C for serialization.
//
//   result_index 0 -> C00, 1 -> C01, ..., N-1 -> C0(N-1), N -> C10, ...
//   row = result_index / N, col = result_index % N (row-major)
//   byte_sel 0 = low byte, 1 = high byte
// ---------------------------------------------------------------------------
module result_mux #(
    parameter int N  = 4,
    parameter int W  = 16,
    parameter int IW = 4
) (
    input  logic [N*N*W-1:0] c_flat,
    input  logic [IW-1:0]    result_index,
    input  logic             byte_sel,
    output logic [7:0]       data
);

    logic [W-1:0] element;

    assign element = c_flat[result_index*W +: W];
    assign data    = byte_sel ? element[15:8] : element[7:0];

endmodule

`default_nettype wire
