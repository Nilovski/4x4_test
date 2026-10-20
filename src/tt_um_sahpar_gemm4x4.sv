`default_nettype none

// ---------------------------------------------------------------------------
// tt_um_sahpar_gemm4x4 -- Tiny Tapeout top level for the 4x4 BF16
// output-stationary systolic GEMM.
//
//   host bytes -> rx_byte_interface -> command_decoder -> operand_buffer A/B
//                                            |                  |
//                                     gemm_controller -> systolic_scheduler
//                                            |                  |
//                                      status_register   systolic_array
//                                            |                  |
//   host bytes <- tx_byte_interface <- command_decoder <- result_mux
//
// Pin map:
//   ui_in[7:0]  = host data in   (RX byte)
//   uo_out[7:0] = host data out  (TX byte)
//   uio[0] = RX_VALID   in   (host asserts when ui_in holds a new byte)
//   uio[1] = RX_READY   out  (always 1)
//   uio[2] = TX_READY   in   (host asserts to accept the byte on uo_out)
//   uio[3] = TX_VALID   out  (chip asserts while uo_out holds an unread byte)
//   uio[4] = BUSY       out  (run in progress; STATUS bit 0)
//   uio[5] = DONE       out  (sticky: a run finished; cleared by the next
//                             GEMM/CLEAR_GEMM; STATUS bit 1). Usable as an
//                             interrupt line.
//   uio[7:6] reserved (inputs)
// ---------------------------------------------------------------------------
module tt_um_sahpar_gemm4x4 (
    input  logic [7:0] ui_in,
    output logic [7:0] uo_out,
    input  logic [7:0] uio_in,
    output logic [7:0] uio_out,
    output logic [7:0] uio_oe,
    input  logic       ena,
    input  logic       clk,
    input  logic       rst_n
);

    localparam int N           = 4;
    localparam int K           = 4;
    localparam int W           = 16;     // BF16
    localparam bit MUL_PIPE    = 1'b0;   // 1: register between MUL and ADD
    localparam int MAC_LATENCY = MUL_PIPE ? 1 : 0;
    localparam int CW          = $clog2(K + 2*(N-1) + MAC_LATENCY);
    localparam int AW          = $clog2(2 * N * K);
    localparam int IW          = $clog2(N * N);

    // ---- byte interfaces -----------------------------------------------
    logic       rx_byte_valid, rx_host_ready;
    logic [7:0] rx_byte;
    logic [7:0] tx_byte;
    logic       tx_byte_valid, tx_ready, tx_host_valid;

    rx_byte_interface u_rx (
        .clk           (clk),
        .rst_n         (rst_n),
        .host_data     (ui_in),
        .host_valid    (uio_in[0]),
        .host_ready    (rx_host_ready),
        .rx_byte       (rx_byte),
        .rx_byte_valid (rx_byte_valid)
    );

    tx_byte_interface u_tx (
        .clk           (clk),
        .rst_n         (rst_n),
        .tx_byte       (tx_byte),
        .tx_byte_valid (tx_byte_valid),
        .tx_ready_o    (tx_ready),
        .host_ready    (uio_in[2]),
        .host_data     (uo_out),
        .host_valid    (tx_host_valid)
    );

    // ---- command decoder -----------------------------------------------
    logic          a_wr_en, b_wr_en;
    logic [AW-1:0] wr_addr;
    logic [7:0]    wr_data;
    logic          gemm_start, gemm_accumulate, gemm_busy, gemm_done;
    logic [IW-1:0] rd_index;
    logic          rd_byte_sel;
    logic [7:0]    rd_data;
    logic [7:0]    status_byte;
    logic          a_load_start, a_load_done, b_load_start, b_load_done;
    logic          error_set, status_read, done_flag;

    command_decoder #(
        .N(N), .K(K), .CHIP_ID(8'hB4)
    ) u_decoder (
        .clk             (clk),
        .rst_n           (rst_n),
        .rx_byte         (rx_byte),
        .rx_byte_valid   (rx_byte_valid & ena),
        .tx_byte         (tx_byte),
        .tx_byte_valid   (tx_byte_valid),
        .tx_ready        (tx_ready),
        .a_wr_en         (a_wr_en),
        .b_wr_en         (b_wr_en),
        .wr_addr         (wr_addr),
        .wr_data         (wr_data),
        .gemm_start      (gemm_start),
        .gemm_accumulate (gemm_accumulate),
        .gemm_busy       (gemm_busy),
        .rd_index        (rd_index),
        .rd_byte_sel     (rd_byte_sel),
        .rd_data         (rd_data),
        .status_byte     (status_byte),
        .a_load_start    (a_load_start),
        .a_load_done     (a_load_done),
        .b_load_start    (b_load_start),
        .b_load_done     (b_load_done),
        .error_set       (error_set),
        .status_read     (status_read)
    );

    status_register u_status (
        .clk          (clk),
        .rst_n        (rst_n),
        .busy         (gemm_busy),
        .run_start    (gemm_start),
        .run_done     (gemm_done),
        .a_load_start (a_load_start),
        .a_load_done  (a_load_done),
        .b_load_start (b_load_start),
        .b_load_done  (b_load_done),
        .error_set    (error_set),
        .status_read  (status_read),
        .status       (status_byte),
        .done_flag    (done_flag)
    );

    // ---- operand buffers -----------------------------------------------
    logic [N*K*W-1:0] a_flat;
    logic [K*N*W-1:0] b_flat;

    operand_buffer #(.NUM_ELEMS(N*K), .W(W), .AW(AW)) u_buf_a (
        .clk     (clk),
        .rst_n   (rst_n),
        .wr_en   (a_wr_en),
        .wr_addr (wr_addr),
        .wr_data (wr_data),
        .flat    (a_flat)
    );

    operand_buffer #(.NUM_ELEMS(K*N), .W(W), .AW(AW)) u_buf_b (
        .clk     (clk),
        .rst_n   (rst_n),
        .wr_en   (b_wr_en),
        .wr_addr (wr_addr),
        .wr_data (wr_data),
        .flat    (b_flat)
    );

    // ---- compute -------------------------------------------------------
    logic          clear_acc, compute_enable;
    logic [CW-1:0] compute_cycle;

    gemm_controller #(
        .N(N), .K(K), .MAC_LATENCY(MAC_LATENCY), .CW(CW)
    ) u_ctrl (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (gemm_start),
        .accumulate     (gemm_accumulate),
        .clear_acc      (clear_acc),
        .compute_enable (compute_enable),
        .compute_cycle  (compute_cycle),
        .busy           (gemm_busy),
        .done           (gemm_done)
    );

    logic [N*W-1:0]   a_west, b_north;
    logic [N-1:0]     a_west_valid, b_north_valid;
    logic [N*N*W-1:0] c_flat;

    systolic_scheduler #(.N(N), .K(K), .W(W), .CW(CW)) u_sched (
        .compute_enable (compute_enable),
        .compute_cycle  (compute_cycle),
        .a_flat         (a_flat),
        .b_flat         (b_flat),
        .a_west         (a_west),
        .a_west_valid   (a_west_valid),
        .b_north        (b_north),
        .b_north_valid  (b_north_valid)
    );

    systolic_array #(.N(N), .W(W), .MUL_PIPE(MUL_PIPE)) u_array (
        .clk           (clk),
        .rst_n         (rst_n),
        .clear_acc     (clear_acc),
        .a_west        (a_west),
        .a_west_valid  (a_west_valid),
        .b_north       (b_north),
        .b_north_valid (b_north_valid),
        .c_flat        (c_flat)
    );

    result_mux #(.N(N), .W(W), .IW(IW)) u_result (
        .c_flat       (c_flat),
        .result_index (rd_index),
        .byte_sel     (rd_byte_sel),
        .data         (rd_data)
    );

    // ---- pins ----------------------------------------------------------
    assign uio_out = {2'b0, done_flag, gemm_busy, tx_host_valid, 1'b0, rx_host_ready, 1'b0};
    assign uio_oe  = 8'b0011_1010;   // [5]=DONE, [4]=BUSY, [3]=TX_VALID, [1]=RX_READY

    wire _unused = &{uio_in[7:3], uio_in[1], 1'b0};

endmodule

`default_nettype wire
