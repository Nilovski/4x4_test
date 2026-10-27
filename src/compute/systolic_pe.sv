`default_nettype none

// ---------------------------------------------------------------------------
// systolic_pe -- one output-stationary processing element.
//
//              A_IN                      B_IN
//                |                         |
//   beat ->   A_REG ------> A_OUT        B_REG ------> B_OUT
//                |                         |
//                +--------> row MAC <------+          (shared, multi-cycle)
//                              |
//                           ACC_REG  (C[i][j], stays in the PE)
//
// On each beat the PE latches A from the west and B from the north and
// passes the previous ones on (A east, B south). During the beat its row's
// BF16 MAC engine computes ACC = ACC + A_REG x B_REG for every PE whose
// operands are valid and writes the result back through acc_we/acc_wdata.
// A_REG and B_REG are stable for the whole beat.
// ---------------------------------------------------------------------------
module systolic_pe #(
    parameter int W = 16
) (
    input  logic         clk,
    input  logic         rst_n,

    input  logic         beat,          // latch new operands (one-cycle pulse)
    input  logic         clear_acc,

    input  logic [W-1:0] a_in,
    input  logic [W-1:0] b_in,
    input  logic         a_valid_in,
    input  logic         b_valid_in,

    output logic [W-1:0] a_out,         // = A_REG
    output logic [W-1:0] b_out,         // = B_REG
    output logic         a_valid_out,
    output logic         b_valid_out,

    output logic         mac_valid,     // A_REG and B_REG both valid this beat

    input  logic         acc_we,        // from the row MAC engine
    input  logic [W-1:0] acc_wdata,
    output logic [W-1:0] acc_out
);

    // operand registers: no reset needed, the valid bits qualify them
    always_ff @(posedge clk) begin
        if (beat) begin
            a_out <= a_in;
            b_out <= b_in;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            a_valid_out <= 1'b0;
            b_valid_out <= 1'b0;
        end else if (beat) begin
            a_valid_out <= a_valid_in;
            b_valid_out <= b_valid_in;
        end
    end

    assign mac_valid = a_valid_out && b_valid_out;

    always_ff @(posedge clk) begin
        if (!rst_n)         acc_out <= '0;
        else if (clear_acc) acc_out <= '0;
        else if (acc_we)    acc_out <= acc_wdata;
    end

endmodule

`default_nettype wire
