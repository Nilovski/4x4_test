`default_nettype none

// ---------------------------------------------------------------------------
// systolic_scheduler -- skewed edge feed for the output-stationary array.
//
// PE(i,j) must see A[i][k] and B[k][j] on the same cycle. Each hop costs one
// cycle, so on compute cycle t:
//
//   row r    gets A[r][k] with k = t - r, valid when 0 <= k < K
//   column c gets B[k][c] with k = t - c, valid when 0 <= k < K
//
// and both operands reach PE(i,j) on cycle k + i + j. Example (N = K = 4):
//
//   t=0: row0=A00                 col0=B00
//   t=1: row0=A01 row1=A10        col0=B10 col1=B01
//   t=2: row0=A02 row1=A11 row2=A20  col0=B20 col1=B11 col2=B02
//
// A is N x K (row-major, a_flat[i*K + k]); B is K x N (b_flat[k*N + j]).
// ---------------------------------------------------------------------------
module systolic_scheduler #(
    parameter int N  = 4,
    parameter int K  = 4,
    parameter int W  = 16,
    parameter int CW = 4
) (
    input  logic             compute_enable,
    input  logic [CW-1:0]    compute_cycle,
    input  logic [N*K*W-1:0] a_flat,
    input  logic [K*N*W-1:0] b_flat,

    output logic [N*W-1:0]   a_west,
    output logic [N-1:0]     a_west_valid,
    output logic [N*W-1:0]   b_north,
    output logic [N-1:0]     b_north_valid
);

    always_comb begin
        a_west        = '0;
        a_west_valid  = '0;
        b_north       = '0;
        b_north_valid = '0;
        if (compute_enable) begin
            for (int r = 0; r < N; r++)
                for (int k = 0; k < K; k++)
                    if (compute_cycle == CW'(k + r)) begin
                        a_west[r*W +: W] = a_flat[(r*K + k)*W +: W];
                        a_west_valid[r]  = 1'b1;
                    end
            for (int c = 0; c < N; c++)
                for (int k = 0; k < K; k++)
                    if (compute_cycle == CW'(k + c)) begin
                        b_north[c*W +: W] = b_flat[(k*N + c)*W +: W];
                        b_north_valid[c]  = 1'b1;
                    end
        end
    end

endmodule

`default_nettype wire
