`default_nettype none

// ---------------------------------------------------------------------------
// bf16_mac_seq -- multi-cycle BF16 multiply-accumulate:
//
//     result = acc + (a x b)        (BF16 multiply, then BF16 add)
//
// Bit-exact with mac() in test/bf16_model.py, i.e. with bf16_multiplier
// followed by bf16_adder: the product is rounded to BF16 first, then added.
//   * Subnormal inputs read as zero; results below the normal range flush
//     to signed zero.
//   * Round to nearest, ties to even.
//   * NaN in, inf x 0 or inf - inf gives the canonical quiet NaN 0x7FC0.
//
// Instead of an 8x8 array multiplier, barrel shifters and a leading-zero
// counter, it uses small iterative hardware:
//   * multiply: 8-step shift-add (one 8-bit adder)
//   * align:    the smaller operand shifts right one bit per cycle
//   * normalize: one bit per cycle
//   * one shared round-to-nearest-even unit for both the product and the sum
//
// Worst case 27 cycles from go to done (see MAC_CYCLES_MAX in the PE).
//
// a, b and acc must stay stable from go until done.
//
// Work register R (17 bits) holds:
//   multiply: R[15:8] partial sum H, R[7:0] multiplier Q; ends as the
//             16-bit significand product
//   add:      R[16] carry, R[15:5] = {hidden, 7 mantissa, G, R, S}, R[4:0] 0
// In both cases rounding reads mant = R[15:8], guard = R[7], sticky = |R[6:0].
// ---------------------------------------------------------------------------
module bf16_mac_seq (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        go,        // one-cycle start pulse
    input  logic [15:0] a,
    input  logic [15:0] b,
    input  logic [15:0] acc,
    output logic        busy,
    output logic        done,      // one-cycle pulse, result valid with it
    output logic [15:0] result
);

    localparam logic [15:0] QNAN = 16'h7FC0;

    typedef enum logic [3:0] {
        S_IDLE,
        S_MUL,
        S_MNORM,
        S_MROUND,
        S_ASETUP,
        S_ALIGN,
        S_ADDSUB,
        S_ANORM,
        S_AROUND
    } state_t;

    state_t             st;
    logic        [16:0] R;
    logic        [10:0] S;        // aligned smaller significand {1, m, G, R, S}
    logic signed [9:0]  E;
    logic        [3:0]  cnt;
    logic        [15:0] prod;     // a x b, rounded to BF16
    logic               big_s;
    logic               eff_sub;

    // ---- field decode -----------------------------------------------------
    function automatic logic f_zero(input logic [15:0] x);
        f_zero = (x[14:7] == 8'h00);
    endfunction
    function automatic logic f_inf(input logic [15:0] x);
        f_inf = (x[14:7] == 8'hFF) && (x[6:0] == 7'd0);
    endfunction
    function automatic logic f_nan(input logic [15:0] x);
        f_nan = (x[14:7] == 8'hFF) && (x[6:0] != 7'd0);
    endfunction

    logic p_s;
    assign p_s = a[15] ^ b[15];

    // ---- shared round + pack (product and sum) ---------------------------
    logic              rnd_inc;
    logic        [8:0] mant_r;
    logic signed [9:0] e_r;
    logic        [6:0] mant7;
    logic              pk_s;
    logic       [15:0] packed_v;

    assign rnd_inc = R[7] & ((|R[6:0]) | R[8]);
    assign mant_r  = {1'b0, R[15:8]} + {8'd0, rnd_inc};

    // One exponent adder, E + e_op, shared by every exponent update:
    //   MUL (first two steps): + exp(b), then - 127
    //   MNORM / ANORM:         + 1 or - 1
    //   MROUND / AROUND:       + rounding carry (result used by pack)
    logic signed [9:0] e_op, e_sum;
    always_comb begin
        e_op = 10'sd0;
        case (st)
            S_MUL:              e_op = (cnt == 4'd7) ? $signed({2'b00, b[14:7]})
                                     : (cnt == 4'd6) ? -10'sd127 : 10'sd0;
            S_MNORM:            e_op = 10'sd1;
            S_ANORM:            e_op = R[16] ? 10'sd1 : -10'sd1;
            S_MROUND, S_AROUND: e_op = $signed({9'd0, mant_r[8]});
            default:            e_op = 10'sd0;
        endcase
    end
    assign e_sum = E + e_op;
    assign e_r   = e_sum;
    assign mant7   = mant_r[8] ? 7'd0 : mant_r[6:0];    // 1.111.. + 1 -> 1.000..
    assign pk_s    = (st == S_MROUND) ? p_s : big_s;

    always_comb begin
        if (e_r >= 10'sd255)    packed_v = {pk_s, 8'hFF, 7'd0};
        else if (e_r <= 10'sd0) packed_v = {pk_s, 15'd0};
        else                    packed_v = {pk_s, e_r[7:0], mant7};
    end

    // ---- multiply step ------------------------------------------------------
    logic [8:0] mul_sum;
    assign mul_sum = {1'b0, R[15:8]} + (R[0] ? {1'b0, 1'b1, a[6:0]} : 9'd0);

    // ---- add setup: order so that |big| >= |small| ---------------------------
    logic       swap;
    logic [7:0] x_e, y_e, bg_e, sm_e, d;
    assign x_e  = acc[14:7];
    assign y_e  = prod[14:7];
    assign swap = acc[14:0] < prod[14:0];
    assign bg_e = swap ? y_e : x_e;
    assign sm_e = swap ? x_e : y_e;
    assign d    = bg_e - sm_e;

    // ---- add / subtract ---------------------------------------------------------
    logic [11:0] addsub;
    assign addsub = {1'b0, R[15:5]} + ({1'b0, S} ^ {12{eff_sub}}) + {11'd0, eff_sub};

    // ---- FSM ---------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            st     <= S_IDLE;
            done   <= 1'b0;
            result <= '0;
        end else begin
            done <= 1'b0;
            case (st)
                S_IDLE: if (go) begin
                    if (f_nan(a) || f_nan(b)) begin
                        prod <= QNAN;
                        st   <= S_ASETUP;
                    end else if (f_inf(a) || f_inf(b)) begin
                        prod <= (f_zero(a) || f_zero(b)) ? QNAN : {p_s, 8'hFF, 7'd0};
                        st   <= S_ASETUP;
                    end else if (f_zero(a) || f_zero(b)) begin
                        prod <= {p_s, 15'd0};
                        st   <= S_ASETUP;
                    end else begin
                        R   <= {9'd0, 1'b1, b[6:0]};
                        E   <= $signed({2'b00, a[14:7]});
                        cnt <= 4'd7;
                        st  <= S_MUL;
                    end
                end

                // H:Q <- (H + (Q[0] ? A : 0)) : Q >> 1, eight times
                S_MUL: begin
                    R[15:0] <= {mul_sum, R[7:1]};
                    E       <= e_sum;
                    cnt     <= cnt - 4'd1;
                    if (cnt == 4'd0) st <= S_MNORM;
                end

                // product is 1x.xx (bump exponent) or 01.xx (shift up)
                S_MNORM: begin
                    if (R[15]) E       <= e_sum;
                    else       R[15:0] <= {R[14:0], 1'b0};
                    st <= S_MROUND;
                end

                S_MROUND: begin
                    prod <= packed_v;
                    st   <= S_ASETUP;
                end

                S_ASETUP: begin
                    if (f_nan(acc) || f_nan(prod)) begin
                        result <= QNAN;
                        done   <= 1'b1;
                        st     <= S_IDLE;
                    end else if (f_inf(acc) && f_inf(prod)) begin
                        result <= (acc[15] == prod[15]) ? acc : QNAN;
                        done   <= 1'b1;
                        st     <= S_IDLE;
                    end else if (f_inf(acc)) begin
                        result <= acc;
                        done   <= 1'b1;
                        st     <= S_IDLE;
                    end else if (f_inf(prod)) begin
                        result <= prod;
                        done   <= 1'b1;
                        st     <= S_IDLE;
                    end else if (f_zero(acc) && f_zero(prod)) begin
                        result <= {acc[15] & prod[15], 15'd0};
                        done   <= 1'b1;
                        st     <= S_IDLE;
                    end else if (f_zero(acc)) begin
                        result <= prod;
                        done   <= 1'b1;
                        st     <= S_IDLE;
                    end else if (f_zero(prod)) begin
                        result <= acc;
                        done   <= 1'b1;
                        st     <= S_IDLE;
                    end else begin
                        R       <= {1'b0, 1'b1, (swap ? prod[6:0] : acc[6:0]), 8'd0};
                        S       <= {1'b1, (swap ? acc[6:0] : prod[6:0]), 3'b000};
                        E       <= $signed({2'b00, bg_e});
                        big_s   <= swap ? prod[15] : acc[15];
                        eff_sub <= acc[15] ^ prod[15];
                        cnt     <= (d >= 8'd11) ? 4'd11 : d[3:0];
                        st      <= (d == 8'd0) ? S_ADDSUB : S_ALIGN;
                    end
                end

                // shift right one bit per cycle, keeping a sticky LSB
                S_ALIGN: begin
                    S   <= {1'b0, S[10:1]} | {10'd0, S[0]};
                    cnt <= cnt - 4'd1;
                    if (cnt == 4'd1) st <= S_ADDSUB;
                end

                S_ADDSUB: begin
                    R[16:5] <= addsub;
                    if (eff_sub && addsub == 12'd0) begin
                        result <= 16'h0000;     // x + (-x) = +0
                        done   <= 1'b1;
                        st     <= S_IDLE;
                    end else begin
                        st <= S_ANORM;
                    end
                end

                S_ANORM: begin
                    if (R[16]) begin                 // carry: shift right
                        R <= {1'b0, R[16:1]};
                        E <= e_sum;
                        st <= S_AROUND;
                    end else if (!R[15]) begin       // cancellation: shift left
                        R <= {R[15:0], 1'b0};
                        E <= e_sum;
                    end else begin
                        st <= S_AROUND;
                    end
                end

                S_AROUND: begin
                    result <= packed_v;
                    done   <= 1'b1;
                    st     <= S_IDLE;
                end

                default: st <= S_IDLE;
            endcase
        end
    end

    assign busy = (st != S_IDLE);

endmodule

`default_nettype wire
