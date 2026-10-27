`default_nettype none

// ---------------------------------------------------------------------------
// systolic_array -- N x N output-stationary BF16 systolic array.
//
//              b_north[0]  b_north[1]
//                  |           |
//   a_west[0] -> PE(0,0) --> PE(0,1) --> ...     row 0 MAC engine
//                  |           |
//   a_west[1] -> PE(1,0) --> PE(1,1) --> ...     row 1 MAC engine
//                  |           |
//
// A moves east and B moves south one PE per beat; C[i][j] stays in PE(i,j).
// Each row has one multi-cycle BF16 MAC (pe_row_engine) that updates the
// row's accumulators during the beat.
// ---------------------------------------------------------------------------
module systolic_array #(
    parameter int N = 4,
    parameter int W = 16
) (
    input  logic             clk,
    input  logic             rst_n,
    input  logic             beat,
    input  logic             clear_acc,
    input  logic [N*W-1:0]   a_west,         // a_west[i]  -> row i
    input  logic [N-1:0]     a_west_valid,
    input  logic [N*W-1:0]   b_north,        // b_north[j] -> column j
    input  logic [N-1:0]     b_north_valid,
    output logic [N*N*W-1:0] c_flat,         // c_flat[i*N+j] = C[i][j]
    output logic             engines_busy
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

    // engines start one cycle after the beat, when the PE registers hold
    // the new operands
    logic start;
    always_ff @(posedge clk) begin
        if (!rst_n) start <= 1'b0;
        else        start <= beat;
    end

    logic [N-1:0] row_busy;

    for (genvar i = 0; i < N; i++) begin : g_row
        logic [N-1:0]   valid, we;
        logic [N*W-1:0] pa, pb, pacc;
        logic [W-1:0]   wdata;

        for (genvar j = 0; j < N; j++) begin : g_col
            systolic_pe #(.W(W)) u_pe (
                .clk         (clk),
                .rst_n       (rst_n),
                .beat        (beat),
                .clear_acc   (clear_acc),
                .a_in        (a_h[i][j]),
                .b_in        (b_v[i][j]),
                .a_valid_in  (av_h[i][j]),
                .b_valid_in  (bv_v[i][j]),
                .a_out       (a_h[i][j+1]),
                .b_out       (b_v[i+1][j]),
                .a_valid_out (av_h[i][j+1]),
                .b_valid_out (bv_v[i+1][j]),
                .mac_valid   (valid[j]),
                .acc_we      (we[j]),
                .acc_wdata   (wdata),
                .acc_out     (c_flat[(i*N + j)*W +: W])
            );
            assign pa[j*W +: W]   = a_h[i][j+1];
            assign pb[j*W +: W]   = b_v[i+1][j];
            assign pacc[j*W +: W] = c_flat[(i*N + j)*W +: W];
        end

        pe_row_engine #(.N(N), .W(W)) u_engine (
            .clk       (clk),
            .rst_n     (rst_n),
            .start     (start),
            .pe_valid  (valid),
            .pe_a      (pa),
            .pe_b      (pb),
            .pe_acc    (pacc),
            .acc_we    (we),
            .acc_wdata (wdata),
            .busy      (row_busy[i])
        );
    end

    assign engines_busy = |row_busy | start;

    logic _unused;
    always_comb begin
        _unused = 1'b0;
        for (int i = 0; i < N; i++)
            _unused = _unused | (&a_h[i][N]) | av_h[i][N] | (&b_v[N][i]) | bv_v[N][i];
    end

endmodule

`default_nettype wire
