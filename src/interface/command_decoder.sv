`default_nettype none

// ---------------------------------------------------------------------------
// command_decoder -- host byte protocol.
//
//   0x00  NOP
//   0x10  LOAD_A          + 2*N*K bytes: A row-major, each element low, high
//   0x11  LOAD_B          + 2*K*N bytes: B row-major, each element low, high
//   0x20  GEMM            C += A x B   (ignored while busy)
//   0x21  CLEAR_GEMM      C  = A x B   (ignored while busy)
//   0x30  READ_C          replies 2*N*N bytes: C00_L, C00_H, ..., C33_H
//   0x31  READ_C_ELEMENT  + 1 index byte (row*N + col); replies C_L, C_H
//   0x40  STATUS          replies the status byte (see status_register)
//   0x41  ID              replies CHIP_ID
//
// Replies go out through tx_byte_interface one byte at a time and respect
// TX_READY back-pressure. Bytes received while a reply is still being sent
// are discarded.
//
// The error flag is raised by: an unknown opcode, GEMM/CLEAR_GEMM or LOAD
// while busy, READ_C/READ_C_ELEMENT while busy, an out-of-range element
// index (the index wraps), or a byte received during a reply. A LOAD sent
// while busy has its data bytes consumed and discarded, so the stored
// operands stay intact and the byte stream stays in sync.
// ---------------------------------------------------------------------------
module command_decoder #(
    parameter int         N       = 4,
    parameter int         K       = 4,
    parameter logic [7:0] CHIP_ID = 8'hB4,
    // derived widths; leave at their defaults
    parameter int         AW      = $clog2(2 * N * K),   // operand byte address
    parameter int         IW      = $clog2(N * N)        // C element index
) (
    input  logic        clk,
    input  logic        rst_n,

    // rx_byte_interface
    input  logic [7:0]  rx_byte,
    input  logic        rx_byte_valid,

    // tx_byte_interface
    output logic [7:0]  tx_byte,
    output logic        tx_byte_valid,
    input  logic        tx_ready,

    // operand buffers
    output logic        a_wr_en,
    output logic        b_wr_en,
    output logic [AW-1:0] wr_addr,
    output logic [7:0]  wr_data,

    // gemm_controller
    output logic        gemm_start,
    output logic        gemm_accumulate,
    input  logic        gemm_busy,

    // result_mux
    output logic [IW-1:0] rd_index,
    output logic        rd_byte_sel,
    input  logic [7:0]  rd_data,

    // status_register
    input  logic [7:0]  status_byte,
    output logic        a_load_start,
    output logic        a_load_done,
    output logic        b_load_start,
    output logic        b_load_done,
    output logic        error_set,
    output logic        status_read
);

    localparam int NUM_A_BYTES = 2 * N * K;
    localparam int NUM_B_BYTES = 2 * K * N;
    localparam int NUM_C_ELEMS = N * N;
    localparam int NUM_C_BYTES = 2 * NUM_C_ELEMS;
    localparam int PW = IW + 1;                 // byte pointer into C
    localparam int LW = $clog2(NUM_C_BYTES + 1);

    localparam logic [7:0] OP_NOP            = 8'h00;
    localparam logic [7:0] OP_LOAD_A         = 8'h10;
    localparam logic [7:0] OP_LOAD_B         = 8'h11;
    localparam logic [7:0] OP_GEMM           = 8'h20;
    localparam logic [7:0] OP_CLEAR_GEMM     = 8'h21;
    localparam logic [7:0] OP_READ_C         = 8'h30;
    localparam logic [7:0] OP_READ_C_ELEMENT = 8'h31;
    localparam logic [7:0] OP_STATUS         = 8'h40;
    localparam logic [7:0] OP_ID             = 8'h41;

    typedef enum logic [2:0] {
        S_IDLE,
        S_LOAD_A,
        S_LOAD_B,
        S_READ_ADDR,
        S_TX
    } state_t;

    state_t        state;
    logic [AW-1:0] ld_idx;
    logic          ld_drop;      // LOAD started while busy: swallow its data
    logic [PW-1:0] tx_ptr;       // {element, byte} of the next C byte
    logic [LW-1:0] tx_left;      // bytes still to send
    logic          tx_from_c;    // reply source: C (1) or reply_reg (0)
    logic [7:0]    reply_reg;

    logic idle_cmd;
    assign idle_cmd = (state == S_IDLE) && rx_byte_valid;

    // ---- combinational strobes ---------------------------------------------
    assign wr_addr = ld_idx;
    assign wr_data = rx_byte;
    assign a_wr_en = (state == S_LOAD_A) && rx_byte_valid && !ld_drop;
    assign b_wr_en = (state == S_LOAD_B) && rx_byte_valid && !ld_drop;

    assign a_load_start = idle_cmd && (rx_byte == OP_LOAD_A) && !gemm_busy;
    assign b_load_start = idle_cmd && (rx_byte == OP_LOAD_B) && !gemm_busy;
    assign a_load_done  = a_wr_en && (ld_idx == AW'(NUM_A_BYTES - 1));
    assign b_load_done  = b_wr_en && (ld_idx == AW'(NUM_B_BYTES - 1));

    assign gemm_start      = idle_cmd && ((rx_byte == OP_GEMM) || (rx_byte == OP_CLEAR_GEMM))
                             && !gemm_busy;
    assign gemm_accumulate = (rx_byte == OP_GEMM);
    assign status_read     = idle_cmd && (rx_byte == OP_STATUS);

    assign rd_index    = tx_ptr[PW-1:1];
    assign rd_byte_sel = tx_ptr[0];

    always_comb begin
        error_set = 1'b0;
        case (state)
            S_IDLE:
                if (rx_byte_valid)
                    case (rx_byte)
                        OP_NOP, OP_STATUS, OP_ID: ;
                        OP_LOAD_A, OP_LOAD_B, OP_GEMM, OP_CLEAR_GEMM,
                        OP_READ_C:         error_set = gemm_busy;
                        OP_READ_C_ELEMENT: ;
                        default:           error_set = 1'b1;
                    endcase
            S_READ_ADDR:
                if (rx_byte_valid)
                    error_set = gemm_busy || (rx_byte >= 8'(NUM_C_ELEMS));
            S_TX:
                error_set = rx_byte_valid;
            default: ;
        endcase
    end

    // ---- sequencing --------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            ld_idx        <= '0;
            ld_drop       <= 1'b0;
            tx_ptr        <= '0;
            tx_left       <= '0;
            tx_from_c     <= 1'b0;
            reply_reg     <= '0;
            tx_byte       <= '0;
            tx_byte_valid <= 1'b0;
        end else begin
            tx_byte_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (rx_byte_valid) begin
                        case (rx_byte)
                            OP_ID: begin
                                reply_reg <= CHIP_ID;
                                tx_from_c <= 1'b0;
                                tx_left   <= LW'(1);
                                state     <= S_TX;
                            end
                            OP_STATUS: begin
                                reply_reg <= status_byte;
                                tx_from_c <= 1'b0;
                                tx_left   <= LW'(1);
                                state     <= S_TX;
                            end
                            OP_LOAD_A: begin
                                ld_idx  <= '0;
                                ld_drop <= gemm_busy;
                                state   <= S_LOAD_A;
                            end
                            OP_LOAD_B: begin
                                ld_idx  <= '0;
                                ld_drop <= gemm_busy;
                                state   <= S_LOAD_B;
                            end
                            OP_READ_C: begin
                                tx_ptr    <= '0;
                                tx_from_c <= 1'b1;
                                tx_left   <= LW'(NUM_C_BYTES);
                                state     <= S_TX;
                            end
                            OP_READ_C_ELEMENT: state <= S_READ_ADDR;
                            default: ;
                        endcase
                    end
                end

                S_LOAD_A: begin
                    if (rx_byte_valid) begin
                        if (ld_idx == AW'(NUM_A_BYTES - 1)) state <= S_IDLE;
                        else                                ld_idx <= ld_idx + AW'(1);
                    end
                end

                S_LOAD_B: begin
                    if (rx_byte_valid) begin
                        if (ld_idx == AW'(NUM_B_BYTES - 1)) state <= S_IDLE;
                        else                                ld_idx <= ld_idx + AW'(1);
                    end
                end

                S_READ_ADDR: begin
                    if (rx_byte_valid) begin
                        tx_ptr    <= {rx_byte[IW-1:0], 1'b0};
                        tx_from_c <= 1'b1;
                        tx_left   <= LW'(2);
                        state     <= S_TX;
                    end
                end

                S_TX: begin
                    // tx_ready drops one cycle after a byte is handed over, so
                    // also wait out the cycle in which tx_byte_valid is high.
                    if (tx_ready && !tx_byte_valid) begin
                        tx_byte       <= tx_from_c ? rd_data : reply_reg;
                        tx_byte_valid <= 1'b1;
                        tx_ptr        <= tx_ptr + PW'(1);
                        tx_left       <= tx_left - LW'(1);
                        if (tx_left == LW'(1)) state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // index bits above the valid range are ignored (the index wraps)
    wire _unused = &{rx_byte[7:IW], 1'b0};

endmodule

`default_nettype wire
