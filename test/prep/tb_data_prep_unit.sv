`timescale 1ns/1ps
`default_nettype none
// Self-checking test for data_prep_unit. Loads random A/B over the byte
// interface, starts a feed and checks every lane output against the skew
// schedule. Feeds twice to check the lanes return to their original order.
// Run with -P tb.N=<n> -P tb.K=<k>.
module tb;
    parameter int N = 4;
    parameter int K = 4;
    localparam int W  = 16;
    localparam int AW = $clog2(2 * N * K);

    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    logic a_wr_en = 0, b_wr_en = 0, start = 0;
    logic [AW-1:0] wr_addr = 0;
    logic [7:0] wr_data = 0;
    logic feeding, feed_done;
    logic [N*W-1:0] a_west, b_north;
    logic [N-1:0] a_west_valid, b_north_valid;

    data_prep_unit #(.N(N), .K(K), .W(W), .AW(AW)) dut (.*);

    logic [W-1:0] A [N*K];   // row-major N x K
    logic [W-1:0] B [K*N];   // row-major K x N
    int errors = 0;

    task automatic load(input bit is_a);
        for (int e = 0; e < N*K; e++)
            for (int h = 0; h < 2; h++) begin
                @(negedge clk);
                a_wr_en = is_a; b_wr_en = !is_a;
                wr_addr = AW'(2*e + h);
                wr_data = is_a ? A[e][8*h +: 8] : B[e][8*h +: 8];
            end
        @(negedge clk); a_wr_en = 0; b_wr_en = 0;
    endtask

    task automatic feed_and_check(input int pass);
        int t;
        @(negedge clk); start = 1;
        for (t = 0; t < K + N - 1; t++) begin
            #1;
            if (!feeding) begin $display("pass %0d t=%0d: feeding low", pass, t); errors++; end
            if (feed_done !== (t == K + N - 2)) begin
                $display("pass %0d t=%0d: feed_done=%b", pass, t, feed_done); errors++;
            end
            for (int i = 0; i < N; i++) begin
                bit act = (t >= i) && (t < i + K);
                logic [W-1:0] ea = act ? A[i*K + (t-i)] : '0;
                logic [W-1:0] eb = act ? B[(t-i)*N + i] : '0;
                if (a_west_valid[i] !== act || a_west[i*W +: W] !== ea) begin
                    $display("pass %0d t=%0d row %0d: got %h v%b exp %h v%b", pass, t, i,
                             a_west[i*W +: W], a_west_valid[i], ea, act);
                    errors++;
                end
                if (b_north_valid[i] !== act || b_north[i*W +: W] !== eb) begin
                    $display("pass %0d t=%0d col %0d: got %h v%b exp %h v%b", pass, t, i,
                             b_north[i*W +: W], b_north_valid[i], eb, act);
                    errors++;
                end
            end
            @(negedge clk); start = 0;
        end
        #1;
        if (feeding || a_west_valid || b_north_valid) begin
            $display("pass %0d: still feeding after %0d cycles", pass, K + N - 1); errors++;
        end
    endtask

    initial begin
        for (int e = 0; e < N*K; e++) begin A[e] = $urandom; B[e] = $urandom; end
        repeat (3) @(negedge clk);
        rst_n = 1;
        load(1); load(0);
        feed_and_check(0);
        repeat (2) @(negedge clk);
        feed_and_check(1);       // same operands, no reload
        for (int e = 0; e < N*K; e++) begin A[e] = $urandom; B[e] = $urandom; end
        load(1); load(0);
        feed_and_check(2);
        if (errors == 0) $display("PASS N=%0d K=%0d", N, K);
        else             $display("FAIL N=%0d K=%0d: %0d errors", N, K, errors);
        $finish;
    end
endmodule
