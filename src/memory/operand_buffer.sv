`default_nettype none

// ---------------------------------------------------------------------------
// operand_buffer -- byte-addressed register file for one operand matrix.
//
// Holds NUM_ELEMS BF16 values in row-major order. Each element takes two
// host bytes, low byte first, so byte address 2*e is the low byte of
// element e and 2*e + 1 its high byte. Used for both A (A[r][k] at element
// r*K + k) and B (B[k][c] at element k*N + c).
//
// Contents persist until overwritten, so either operand can be reloaded on
// its own.
// ---------------------------------------------------------------------------
module operand_buffer #(
    parameter int NUM_ELEMS = 16,
    parameter int W         = 16,
    parameter int AW        = 5
) (
    input  logic                   clk,
    input  logic                   rst_n,

    input  logic                   wr_en,
    input  logic [AW-1:0]          wr_addr,   // byte address
    input  logic [7:0]             wr_data,

    output logic [NUM_ELEMS*W-1:0] flat       // element e at [e*W +: W]
);

    localparam int NUM_BYTES = NUM_ELEMS * W / 8;

    logic [7:0] mem [NUM_BYTES];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < NUM_BYTES; i++) mem[i] <= '0;
        end else if (wr_en) begin
            mem[wr_addr] <= wr_data;
        end
    end

    always_comb begin
        for (int i = 0; i < NUM_BYTES; i++)
            flat[i*8 +: 8] = mem[i];
    end

endmodule

`default_nettype wire
