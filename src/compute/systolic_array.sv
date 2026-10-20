`default_nettype none

// ---------------------------------------------------------------------------
// systolic_array -- N x N output-stationary mesh of systolic_pe.
//
//              b_north[0]   b_north[1]  ...
//                  |            |
//   a_west[0] --> [00] -------> [01] ---> ...
//                  |            |
//   a_west[1] --> [10] -------> [11] ---> ...
//                  |            |
//
// A enters from the west edge and flows east; B enters from the north edge
// and flows south, each with a valid bit. PE(i,j) accumulates C[i][j] in
// place, so results are read through a mux rather than a drain chain.
// ---------------------------------------------------------------------------
module systolic_array #(
    parameter int N        = 4,
    parameter int W        = 16,
    parameter bit MUL_PIPE = 1'b0
) (
    input  logic             clk,
    input  logic             rst_n,
    input  logic             clear_acc,

    input  logic [N*W-1:0]   a_west,         // a_west[i]  -> row i
    input  logic [N-1:0]     a_west_valid,
    input  logic [N*W-1:0]   b_north,        // b_north[j] -> column j
    input  logic [N-1:0]     b_north_valid,

    output logic [N*N*W-1:0] c_flat          // c_flat[i*N+j] = C[i][j]
);

    logic [W-1:0] a_h  [N][N+1];  // horizontal links (west -> east)
    logic         av_h [N][N+1];
    logic [W-1:0] b_v  [N+1][N];  // vertical links (north -> south)
    logic         bv_v [N+1][N];

    for (genvar i = 0; i < N; i++) begin : g_west_edge
        assign a_h[i][0]  = a_west[i*W +: W];
        assign av_h[i][0] = a_west_valid[i];
    end
    for (genvar j = 0; j < N; j++) begin : g_north_edge
        assign b_v[0][j]  = b_north[j*W +: W];
        assign bv_v[0][j] = b_north_valid[j];
    end

    for (genvar i = 0; i < N; i++) begin : g_row
        for (genvar j = 0; j < N; j++) begin : g_col
            systolic_pe #(
                .W        (W),
                .MUL_PIPE (MUL_PIPE)
            ) u_pe (
                .clk         (clk),
                .rst_n       (rst_n),
                .clear_acc   (clear_acc),
                .a_in        (a_h[i][j]),
                .b_in        (b_v[i][j]),
                .a_valid_in  (av_h[i][j]),
                .b_valid_in  (bv_v[i][j]),
                .a_out       (a_h[i][j+1]),
                .b_out       (b_v[i+1][j]),
                .a_valid_out (av_h[i][j+1]),
                .b_valid_out (bv_v[i+1][j]),
                .acc_out     (c_flat[(i*N + j)*W +: W])
            );
        end
    end

    // The east and south edges leave the array unconnected.
    logic _unused;
    always_comb begin
        _unused = 1'b0;
        for (int i = 0; i < N; i++)
            _unused = _unused | (&a_h[i][N]) | av_h[i][N] | (&b_v[N][i]) | bv_v[N][i];
    end

endmodule

`default_nettype wire
