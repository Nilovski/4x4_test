`default_nettype none

// ---------------------------------------------------------------------------
// bf16_adder -- combinational BF16 + BF16 -> BF16.
//
//   * Subnormal inputs read as zero; results below the normal range flush
//     to signed zero.
//   * Round to nearest, ties to even, using guard/round/sticky bits.
//   * NaN input or inf - inf gives the canonical quiet NaN 0x7FC0.
//   * x + (-x) = +0; (-0) + (-0) = -0.
//
// Bit-exact with add() in test/bf16_model.py.
// ---------------------------------------------------------------------------
module bf16_adder (
    input  logic [15:0] a,
    input  logic [15:0] b,
    output logic [15:0] y
);

    localparam logic [15:0] QNAN = 16'h7FC0;

    logic       sa, sb;
    logic [7:0] ea, eb;
    logic [6:0] ma, mb;

    assign {sa, ea, ma} = a;
    assign {sb, eb, mb} = b;

    logic a_zero, b_zero, a_inf, b_inf, a_nan, b_nan;
    assign a_zero = (ea == 8'h00);
    assign b_zero = (eb == 8'h00);
    assign a_inf  = (ea == 8'hFF) && (ma == 7'd0);
    assign b_inf  = (eb == 8'hFF) && (mb == 7'd0);
    assign a_nan  = (ea == 8'hFF) && (ma != 7'd0);
    assign b_nan  = (eb == 8'hFF) && (mb != 7'd0);

    // ---- order operands so that |big| >= |small| --------------------------
    logic       swap;
    logic       big_s, small_s;
    logic [7:0] big_e, small_e;
    logic [6:0] big_m, small_m;

    assign swap = {ea, ma} < {eb, mb};
    assign {big_s, big_e, big_m}       = swap ? {sb, eb, mb} : {sa, ea, ma};
    assign {small_s, small_e, small_m} = swap ? {sa, ea, ma} : {sb, eb, mb};

    // ---- align: 11-bit significands {hidden, 7 mantissa, G, R, S} ----------
    logic [7:0]  d;
    logic [10:0] big_sig, small_full, small_sig, sticky_mask;

    assign d          = big_e - small_e;
    assign big_sig    = {1'b1, big_m, 3'b000};
    assign small_full = {1'b1, small_m, 3'b000};

    logic in_range;
    assign in_range    = (d < 8'd11);
    assign sticky_mask = in_range ? ((11'd1 << d[3:0]) - 11'd1) : 11'd0;
    assign small_sig   = in_range ? ((small_full >> d[3:0]) | {10'd0, |(small_full & sticky_mask)})
                                  : 11'd1;           // shifted out entirely

    // ---- add / subtract and normalize --------------------------------------
    logic              eff_sub;
    logic       [11:0] sum;
    logic       [10:0] diff, r;
    logic        [3:0] lz;
    logic signed [9:0] e;
    logic        [8:0] mant_r;
    logic              g, sticky;

    assign eff_sub = big_s ^ small_s;
    assign sum     = {1'b0, big_sig} + {1'b0, small_sig};
    assign diff    = big_sig - small_sig;

    // leading-zero count of diff (diff != 0 whenever it is used)
    function automatic logic [3:0] lzc11(input logic [10:0] x);
        lzc11 = 4'd0;
        for (int i = 0; i < 11; i++)
            if (x[i]) lzc11 = 4'(10 - i);
    endfunction

    assign lz = lzc11(diff);

    // normalize: effective add may carry into bit 11 (shift right, keep
    // sticky); effective subtract shifts left by the leading-zero count
    logic signed [9:0] e_norm;
    assign r = !eff_sub ? (sum[11] ? {sum[11:2], sum[1] | sum[0]} : sum[10:0])
                        : (diff << lz);
    assign e_norm = !eff_sub ? ($signed({2'b00, big_e}) + $signed({9'd0, sum[11]}))
                             : ($signed({2'b00, big_e}) - $signed({6'd0, lz}));

    // round to nearest even on r = {1.mmmmmmm, G, R, S}
    assign g      = r[2];
    assign sticky = r[1] | r[0];
    assign mant_r = {1'b0, r[10:3]} + {8'd0, g & (sticky | r[3])};
    assign e      = e_norm + $signed({9'd0, mant_r[8]});

    logic [15:0] y_inf, y_zero, y_norm;
    assign y_inf  = {big_s, 8'hFF, 7'd0};
    assign y_zero = {big_s, 15'd0};
    assign y_norm = {big_s, e[7:0], mant_r[6:0]};

    always_comb begin
        if (a_nan || b_nan)                y = QNAN;
        else if (a_inf && b_inf)           y = (sa == sb) ? a : QNAN;
        else if (a_inf)                    y = a;
        else if (b_inf)                    y = b;
        else if (a_zero && b_zero)         y = {sa & sb, 15'd0};
        else if (a_zero)                   y = b;
        else if (b_zero)                   y = a;
        else if (eff_sub && diff == 11'd0) y = 16'h0000;
        else if (e >= 10'sd255)            y = y_inf;
        else if (e <= 10'sd0)              y = y_zero;
        else                               y = y_norm;
    end

    // mant_r[7] is the hidden bit
    wire _unused = &{mant_r[7], 1'b0};

endmodule

`default_nettype wire
