`default_nettype none

// ---------------------------------------------------------------------------
// gemm_controller -- run sequencing for one N x N x K GEMM.
//
//   IDLE --start--> CLEAR --> COMPUTE --> DRAIN --> DONE --> IDLE
//              \_ accumulate _/
//
//   CLEAR    1 cycle, clear_acc to every PE (skipped when accumulate = 1,
//            so C += A x B).
//   COMPUTE  K + N - 1 cycles; the data prep unit feeds skewed operands.
//   DRAIN    N - 1 + MAC_LATENCY cycles for the last operands to reach
//            PE(N-1,N-1) and leave the MAC pipeline.
//   DONE     1 cycle; done pulses, results are valid from here on.
//
// busy is high in every state except IDLE; start is ignored while busy.
// ---------------------------------------------------------------------------
module gemm_controller #(
    parameter int N           = 4,
    parameter int K           = 4,
    parameter int MAC_LATENCY = 0,
    parameter int CW          = 4
) (
    input  logic          clk,
    input  logic          rst_n,

    input  logic          start,
    input  logic          accumulate,

    output logic          clear_acc,
    output logic          compute_enable,
    output logic [CW-1:0] compute_cycle,
    output logic          busy,
    output logic          done
);

    localparam int COMPUTE_CYCLES = K + N - 1;
    localparam int DRAIN_CYCLES   = N - 1 + MAC_LATENCY;
    localparam int LAST_CYCLE     = COMPUTE_CYCLES + DRAIN_CYCLES - 1;

    typedef enum logic [2:0] {
        S_IDLE,
        S_CLEAR,
        S_COMPUTE,
        S_DRAIN,
        S_DONE
    } state_t;

    state_t state;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            compute_cycle <= '0;
        end else begin
            case (state)
                S_IDLE: begin
                    compute_cycle <= '0;
                    if (start) state <= accumulate ? S_COMPUTE : S_CLEAR;
                end

                S_CLEAR: state <= S_COMPUTE;

                S_COMPUTE: begin
                    compute_cycle <= compute_cycle + CW'(1);
                    if (compute_cycle == CW'(COMPUTE_CYCLES - 1))
                        state <= (DRAIN_CYCLES == 0) ? S_DONE : S_DRAIN;
                end

                S_DRAIN: begin
                    compute_cycle <= compute_cycle + CW'(1);
                    if (compute_cycle == CW'(LAST_CYCLE)) state <= S_DONE;
                end

                S_DONE: state <= S_IDLE;

                default: state <= S_IDLE;
            endcase
        end
    end

    assign clear_acc      = (state == S_CLEAR);
    assign compute_enable = (state == S_COMPUTE);
    assign busy           = (state != S_IDLE);
    assign done           = (state == S_DONE);

endmodule

`default_nettype wire
