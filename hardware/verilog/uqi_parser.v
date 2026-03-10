module uqi_parser #(
    parameter ADDR_WIDTH = 128,
    parameter DATA_WIDTH = 512
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // AXI-Stream Input
    input  wire [DATA_WIDTH-1:0]   s_axis_tdata,
    input  wire                    s_axis_tvalid,
    output wire                    s_axis_tready,

    output reg  [DATA_WIDTH-1:0]   payload_out,
    output reg  [31:0]             phase_target
);

    assign s_axis_tready = 1'b1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            payload_out <= 0;
            phase_target <= 0;
        end else if (s_axis_tvalid) begin
            payload_out <= s_axis_tdata;
            // Extract target phase from header bits (hypothetical mapping)
            phase_target <= s_axis_tdata[31:0];
        end
    end

endmodule
