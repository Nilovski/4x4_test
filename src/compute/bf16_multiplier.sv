`default_nettype none

// ---------------------------------------------------------------------------
// bf16_multiplier -- combinational BF16 x BF16 -> BF16.
//
//   * Subnormal inputs read as zero; results below the normal range flush
//     to signed zero.
//   * Round to nearest, ties to even.
//   * Any NaN input, or inf x 0, gives the canonical quiet NaN 0x7FC0.
//
// Bit-exact with mul() in test/bf16_model.py.
// ---------------------------------------------------------------------------
module bf16_multiplier (
    input  logic [15:0] a,
    input  logic [15:0] b,
    output logic [15:0] y
);

    localparam logic [15:0] QNAN = 16'h7FC0;

    logic       sa, sb, s;
    logic [7:0] ea, eb;
    logic [6:0] ma, mb;

    assign {sa, ea, ma} = a;
    assign {sb, eb, mb} = b;
    assign s = sa ^ sb;

    logic a_zero, b_zero, a_inf, b_inf, a_nan, b_nan;
    assign a_zero = (ea == 8'h00);
    assign b_zero = (eb == 8'h00);
    assign a_inf  = (ea == 8'hFF) && (ma == 7'd0);
    assign b_inf  = (eb == 8'hFF) && (mb == 7'd0);
    assign a_nan  = (ea == 8'hFF) && (ma != 7'd0);
    assign b_nan  = (eb == 8'hFF) && (mb != 7'd0);

    // significand product, 1.xxx * 1.xxx in [1, 4)
    logic [15:0] p;
    assign p = {1'b1, ma} * {1'b1, mb};

    logic              p_hi;
    logic        [7:0] mant;
    logic              g, sticky;
    logic        [8:0] mant_r;
    logic signed [9:0] e;

    // normalize: product is 01.xx (p_hi = 0) or 1x.xx (p_hi = 1)
    assign p_hi   = p[15];
    assign mant   = p_hi ? p[15:8] : p[14:7];
    assign g      = p_hi ? p[7]    : p[6];
    assign sticky = p_hi ? |p[6:0] : |p[5:0];

    // round to nearest even; a carry out of 1.111.. bumps the exponent
    assign mant_r = {1'b0, mant} + {8'd0, g & (sticky | mant[0])};
    assign e      = $signed({2'b00, ea}) + $signed({2'b00, eb}) - 10'sd127
                  + $signed({9'd0, p_hi}) + $signed({9'd0, mant_r[8]});

    logic [15:0] y_inf, y_zero, y_norm;
    assign y_inf  = {s, 8'hFF, 7'd0};
    assign y_zero = {s, 15'd0};
    assign y_norm = {s, e[7:0], mant_r[6:0]};

    always_comb begin
        if (a_nan || b_nan)            y = QNAN;
        else if (a_inf || b_inf)       y = (a_zero || b_zero) ? QNAN : y_inf;
        else if (a_zero || b_zero)     y = y_zero;
        else if (e >= 10'sd255)        y = y_inf;
        else if (e <= 10'sd0)          y = y_zero;
        else                           y = y_norm;
    end

    // mant_r[7] is the hidden bit
    wire _unused = &{mant_r[7], 1'b0};

endmodule

`default_nettype wire
