module arkhe_fpga_top (
    input  wire        clk_100mhz,
    input  wire        reset_n,
    output wire [7:0]  debug_leds,
    output wire        heartbeat
);
    reg [23:0] counter;
    always @(posedge clk_100mhz or negedge reset_n) begin
        if (!reset_n) counter <= 0;
        else counter <= counter + 1;
    end
    assign heartbeat = counter[23];
    assign debug_leds = counter[23:16];
endmodule
