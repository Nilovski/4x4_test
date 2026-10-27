`default_nettype none

// ---------------------------------------------------------------------------
// gemm_controller -- runs one GEMM as a sequence of beats.
//
//   IDLE -> CLEAR -> RUN (NUM_BEATS beats of BEAT cycles) -> DONE -> IDLE
//
// CLEAR zeroes the accumulators (CLEAR_GEMM only; GEMM keeps them) and starts
// the operand feed, so the first beat sees feed index 0. On the first cycle
// of every beat the PEs latch new operands and the feed advances; the row
// MAC engines then work through the beat. NUM_BEATS = K + 2(N-1) covers the
// skew in and the drain to PE(N-1,N-1).
// ---------------------------------------------------------------------------
module gemm_controller #(
    parameter int N    = 4,
    parameter int K    = 4,
    parameter int BEAT = 128      // cycles per beat; >= row engine worst case
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    input  logic accumulate,
    input  logic engines_busy,    // simulation check only
    output logic clear_acc,
    output logic feed_start,
    output logic beat,
    output logic busy,
    output logic done
);

    localparam int NUM_BEATS = K + 2 * (N - 1);
    localparam int CCW       = $clog2(BEAT);
    localparam int BCW       = $clog2(NUM_BEATS);

    typedef enum logic [1:0] {
        S_IDLE,
        S_CLEAR,
        S_RUN,
        S_DONE
    } state_t;

    state_t         state;
    logic           keep_acc;
    logic [CCW-1:0] cc;           // cycle within the beat
    logic [BCW-1:0] bc;           // beat number

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            keep_acc <= 1'b0;
            cc       <= '0;
            bc       <= '0;
        end else begin
            case (state)
                S_IDLE: if (start) begin
                    keep_acc <= accumulate;
                    state    <= S_CLEAR;
                end
                S_CLEAR: begin
                    cc    <= '0;
                    bc    <= '0;
                    state <= S_RUN;
                end
                S_RUN: begin
                    if (cc == CCW'(BEAT - 1)) begin
                        cc <= '0;
                        if (bc == BCW'(NUM_BEATS - 1)) state <= S_DONE;
                        else                           bc    <= bc + BCW'(1);
                    end else begin
                        cc <= cc + CCW'(1);
                    end
                end
                S_DONE:  state <= S_IDLE;
                default: state <= S_IDLE;
            endcase
        end
    end

    assign clear_acc  = (state == S_CLEAR) && !keep_acc;
    assign feed_start = (state == S_CLEAR);
    assign beat       = (state == S_RUN) && (cc == '0);
    assign busy       = (state != S_IDLE);
    assign done       = (state == S_DONE);

`ifndef SYNTHESIS
    // the row MAC engines must be idle when the next beat (or the end) comes
    always_ff @(posedge clk) begin
        if (rst_n && (beat || state == S_DONE) && engines_busy)
            $error("gemm_controller: BEAT=%0d is too short for the MAC engines", BEAT);
    end
`else
    wire _unused = &{engines_busy, 1'b0};
`endif

endmodule

`default_nettype wire
