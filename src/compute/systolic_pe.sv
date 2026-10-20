`default_nettype none

// ---------------------------------------------------------------------------
// systolic_pe -- one output-stationary BF16 processing element.
//
//              b_in / b_valid_in (north)
//                       |
//   a_in / a_valid_in --+--> [A_REG] --> a_out / a_valid_out (east)
//        (west)         |
//                  BF16 MUL --> [P_REG]* --> BF16 ADD <-- ACC_REG
//                       |                       |
//                       |                       +--> ACC_REG --> acc_out
//                       +--> [B_REG] --> b_out / b_valid_out (south)
//
//   * A and B are forwarded east/south with one cycle of latency, together
//     with their valid bits.
//   * When a_valid_in && b_valid_in, ACC <= ACC + a_in * b_in.
//   * clear_acc sets ACC to +0 and takes priority.
//   * MUL_PIPE = 1 inserts the optional P_REG after the multiplier
//     (one extra cycle of MAC latency, shorter critical path).
// ---------------------------------------------------------------------------
module systolic_pe #(
    parameter int W        = 16,
    parameter bit MUL_PIPE = 1'b0
) (
    input  logic         clk,
    input  logic         rst_n,

    input  logic         clear_acc,

    input  logic [W-1:0] a_in,
    input  logic [W-1:0] b_in,
    input  logic         a_valid_in,
    input  logic         b_valid_in,

    output logic [W-1:0] a_out,
    output logic [W-1:0] b_out,
    output logic         a_valid_out,
    output logic         b_valid_out,

    output logic [W-1:0] acc_out
);

    // ---- operand forwarding ------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            a_out       <= '0;
            b_out       <= '0;
            a_valid_out <= 1'b0;
            b_valid_out <= 1'b0;
        end else begin
            a_out       <= a_in;
            b_out       <= b_in;
            a_valid_out <= a_valid_in;
            b_valid_out <= b_valid_in;
        end
    end

    // ---- multiply ----------------------------------------------------------
    logic [W-1:0] prod;
    logic [W-1:0] prod_x;     // product as seen by the adder
    logic         prod_valid;

    bf16_multiplier u_mul (
        .a (a_in),
        .b (b_in),
        .y (prod)
    );

    if (MUL_PIPE) begin : g_mul_pipe
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                prod_x     <= '0;
                prod_valid <= 1'b0;
            end else begin
                prod_x     <= prod;
                prod_valid <= a_valid_in && b_valid_in;
            end
        end
    end else begin : g_mul_comb
        assign prod_x     = prod;
        assign prod_valid = a_valid_in && b_valid_in;
    end

    // ---- accumulate --------------------------------------------------------
    logic [W-1:0] acc;
    logic [W-1:0] sum;

    bf16_adder u_add (
        .a (acc),
        .b (prod_x),
        .y (sum)
    );

    always_ff @(posedge clk) begin
        if (!rst_n)          acc <= '0;
        else if (clear_acc)  acc <= '0;
        else if (prod_valid) acc <= sum;
    end

    assign acc_out = acc;

endmodule

`default_nettype wire
