`default_nettype none

// ---------------------------------------------------------------------------
// data_prep_unit -- operand lane buffers and skewed feed for the array.
//
// Holds A and B as one K-deep buffer per array lane:
//
//   A lane r (west edge, row r):    A[r][0], A[r][1], ..., A[r][K-1]
//   B lane c (north edge, col c):   B[0][c], B[1][c], ..., B[K-1][c]
//
// N = K = 4 gives four 4-deep lanes per operand; N = K = 8 gives eight
// 8-deep lanes. Nothing else changes with the size.
//
// Loading: byte-serial, same addressing as the host protocol. A is sent
// row-major (element e = r*K + k), B row-major (element e = k*N + c), each
// element low byte first, so byte address 2*e is the low byte of e.
//
// Feeding: a one-cycle start pulse begins a feed of K + N - 1 steps; from
// the next cycle the unit presents feed index t = 0, and each step pulse
// (one per beat) advances t. At index t, lane r presents A[r][t-r] (valid
// while r <= t < r + K) and lane c presents B[t-c][c] (valid while
// c <= t < c + K). PE(i,j) then holds A[i][k] and B[k][j] together in beat
// k + i + j.
//
// Each lane is a rotating register: the head is the output, and the lane
// rotates by one on each step while it is valid. After its K valid cycles every lane is
// back in its original order, so the same operands can be fed again (a new
// start, or GEMM with accumulate) without reloading.
//
// Writes are ignored while feeding.
// ---------------------------------------------------------------------------
module data_prep_unit #(
    parameter int N  = 4,
    parameter int K  = 4,
    parameter int W  = 16,
    parameter int AW = $clog2(2 * N * K)
) (
    input  logic           clk,
    input  logic           rst_n,

    // byte loads from the command decoder
    input  logic           a_wr_en,
    input  logic           b_wr_en,
    input  logic [AW-1:0]  wr_addr,     // byte address
    input  logic [7:0]     wr_data,

    // feed control
    input  logic           start,       // one-cycle pulse; ignored while feeding
    input  logic           step,        // advance to the next feed index
    output logic           feeding,     // high while a feed is in progress
    output logic           feed_done,   // high on the step that ends the feed

    // skewed operands to the array edges
    output logic [N*W-1:0] a_west,
    output logic [N-1:0]   a_west_valid,
    output logic [N*W-1:0] b_north,
    output logic [N-1:0]   b_north_valid
);

    localparam int FEED_CYCLES = K + N - 1;
    localparam int TW          = (FEED_CYCLES > 1) ? $clog2(FEED_CYCLES) : 1;

    logic [W-1:0] a_lane [N][K];   // a_lane[r][k] = A[r][k] at rest
    logic [W-1:0] b_lane [N][K];   // b_lane[c][k] = B[k][c] at rest

    // ---- feed counter --------------------------------------------------
    logic          running;
    logic [TW-1:0] t;               // current feed index
    logic          adv;             // a step that moves the feed on

    assign feeding   = running;
    assign adv       = running && step;
    assign feed_done = adv && (t == TW'(FEED_CYCLES - 1));

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            running <= 1'b0;
            t       <= '0;
        end else if (start && !running) begin
            running <= 1'b1;
            t       <= '0;
        end else if (adv) begin
            running <= !feed_done;
            t       <= feed_done ? '0 : t + TW'(1);
        end
    end

    // ---- lane valid windows --------------------------------------------
    logic [N-1:0] a_active, b_active;

    always_comb begin
        for (int i = 0; i < N; i++) begin
            a_active[i] = running && (t >= TW'(i)) && (t < TW'(i + K));
            b_active[i] = a_active[i];   // same window: lane i, cycles i .. i+K-1
        end
    end

    // ---- outputs: lane heads -------------------------------------------
    always_comb begin
        for (int i = 0; i < N; i++) begin
            a_west[i*W +: W]  = a_active[i] ? a_lane[i][0] : '0;
            b_north[i*W +: W] = b_active[i] ? b_lane[i][0] : '0;
        end
    end

    assign a_west_valid  = a_active;
    assign b_north_valid = b_active;

    // ---- load / rotate -------------------------------------------------
    // byte address -> element, half; element -> lane, slot
    logic [AW-2:0] e;
    logic          hi;

    assign e  = wr_addr[AW-1:1];
    assign hi = wr_addr[0];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < N; i++)
                for (int k = 0; k < K; k++) begin
                    a_lane[i][k] <= '0;
                    b_lane[i][k] <= '0;
                end
        end else if (running) begin
            if (step) for (int i = 0; i < N; i++) begin
                if (a_active[i]) begin
                    for (int k = 0; k < K - 1; k++) a_lane[i][k] <= a_lane[i][k+1];
                    a_lane[i][K-1] <= a_lane[i][0];
                end
                if (b_active[i]) begin
                    for (int k = 0; k < K - 1; k++) b_lane[i][k] <= b_lane[i][k+1];
                    b_lane[i][K-1] <= b_lane[i][0];
                end
            end
        end else begin
            for (int i = 0; i < N; i++)
                for (int k = 0; k < K; k++) begin
                    // A element r*K + k -> a_lane[r][k]
                    if (a_wr_en && e == (AW-1)'(i*K + k)) begin
                        if (hi) a_lane[i][k][W-1:8] <= wr_data;
                        else    a_lane[i][k][7:0]   <= wr_data;
                    end
                    // B element k*N + c -> b_lane[c][k]
                    if (b_wr_en && e == (AW-1)'(k*N + i)) begin
                        if (hi) b_lane[i][k][W-1:8] <= wr_data;
                        else    b_lane[i][k][7:0]   <= wr_data;
                    end
                end
        end
    end

endmodule

`default_nettype wire
