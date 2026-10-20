`default_nettype none

// ---------------------------------------------------------------------------
// status_register -- flags reported by the STATUS command.
//
//   bit 0  busy          a run is in progress
//   bit 1  done          a run finished; cleared when the next run starts
//   bit 2  a_loaded      a complete LOAD_A has been received
//   bit 3  b_loaded      a complete LOAD_B has been received
//   bit 4  result_valid  C = result for the currently loaded A and B;
//                        cleared when a run or a LOAD starts
//   bit 5  error         protocol error since the last STATUS read
//                        (cleared by reading STATUS)
//   bit 7:6 reserved, read as 0
// ---------------------------------------------------------------------------
module status_register (
    input  logic       clk,
    input  logic       rst_n,

    input  logic       busy,
    input  logic       run_start,
    input  logic       run_done,
    input  logic       a_load_start,
    input  logic       a_load_done,
    input  logic       b_load_start,
    input  logic       b_load_done,
    input  logic       error_set,
    input  logic       status_read,

    output logic [7:0] status,
    output logic       done_flag
);

    logic a_loaded, b_loaded, result_valid, error;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            done_flag    <= 1'b0;
            a_loaded     <= 1'b0;
            b_loaded     <= 1'b0;
            result_valid <= 1'b0;
            error        <= 1'b0;
        end else begin
            if (run_done)       done_flag <= 1'b1;
            else if (run_start) done_flag <= 1'b0;

            if (a_load_done)       a_loaded <= 1'b1;
            else if (a_load_start) a_loaded <= 1'b0;

            if (b_load_done)       b_loaded <= 1'b1;
            else if (b_load_start) b_loaded <= 1'b0;

            if (run_done)
                result_valid <= 1'b1;
            else if (run_start || a_load_start || b_load_start)
                result_valid <= 1'b0;

            if (error_set)        error <= 1'b1;
            else if (status_read) error <= 1'b0;
        end
    end

    assign status = {2'b00, error, result_valid, b_loaded, a_loaded, done_flag, busy};

endmodule

`default_nettype wire
