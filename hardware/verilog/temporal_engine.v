module temporal_engine #(
    parameter TIMESTAMP_WIDTH = 64
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire [TIMESTAMP_WIDTH-1:0] target_time,
    input  wire        valid,
    input  wire [TIMESTAMP_WIDTH-1:0] current_time,

    output reg         retrocausal,
    output wire        ready
);

    assign ready = 1'b1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            retrocausal <= 0;
        end else if (valid) begin
            retrocausal <= (target_time < current_time);
        end
    end
endmodule
