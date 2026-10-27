`default_nettype none

// ---------------------------------------------------------------------------
// pe_row_engine -- one multi-cycle BF16 MAC shared by the N PEs of a row.
//
// After each beat it visits PE(i,0) .. PE(i,N-1) in order. For every PE
// whose operands are valid it runs ACC = ACC + A x B on bf16_mac_seq and
// writes the result back into that PE's accumulator. PEs without valid
// operands are skipped in one cycle.
//
// Worst case per beat: N * (MAC_CYCLES_MAX + 2) + 1 cycles; the controller's
// beat length must be at least that (checked in simulation).
// ---------------------------------------------------------------------------
module pe_row_engine #(
    parameter int N  = 4,
    parameter int W  = 16,
    parameter int JW = (N > 1) ? $clog2(N) : 1
) (
    input  logic           clk,
    input  logic           rst_n,
    input  logic           start,          // one cycle after a beat
    input  logic [N-1:0]   pe_valid,       // mac_valid of each PE in the row
    input  logic [N*W-1:0] pe_a,           // A_REG of each PE
    input  logic [N*W-1:0] pe_b,           // B_REG of each PE
    input  logic [N*W-1:0] pe_acc,         // ACC_REG of each PE
    output logic [N-1:0]   acc_we,
    output logic [W-1:0]   acc_wdata,
    output logic           busy
);

    typedef enum logic [1:0] {
        Q_IDLE,
        Q_PICK,
        Q_WAIT
    } qstate_t;

    qstate_t       q;
    logic [JW-1:0] j;
    logic          go, mac_busy, mac_done;
    logic [W-1:0]  mac_result;

    assign go = (q == Q_PICK) && pe_valid[j];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            q <= Q_IDLE;
            j <= '0;
        end else begin
            case (q)
                Q_IDLE: if (start) begin
                    j <= '0;
                    q <= Q_PICK;
                end
                Q_PICK: begin
                    if (pe_valid[j])                q <= Q_WAIT;
                    else if (j == JW'(N - 1))       q <= Q_IDLE;
                    else                            j <= j + JW'(1);
                end
                Q_WAIT: if (mac_done) begin
                    if (j == JW'(N - 1)) q <= Q_IDLE;
                    else begin
                        j <= j + JW'(1);
                        q <= Q_PICK;
                    end
                end
                default: q <= Q_IDLE;
            endcase
        end
    end

    bf16_mac_seq u_mac (
        .clk    (clk),
        .rst_n  (rst_n),
        .go     (go),
        .a      (pe_a[j*W +: W]),
        .b      (pe_b[j*W +: W]),
        .acc    (pe_acc[j*W +: W]),
        .busy   (mac_busy),
        .done   (mac_done),
        .result (mac_result)
    );

    always_comb begin
        acc_we = '0;
        acc_we[j] = mac_done;
    end
    assign acc_wdata = mac_result;
    assign busy      = (q != Q_IDLE);

    wire _unused = &{mac_busy, 1'b0};

endmodule

`default_nettype wire
