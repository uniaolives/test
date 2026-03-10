module timechain_anchor #(
    parameter HASH_WIDTH = 256,
    parameter BLOCK_SIZE = 1024
)(
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire [255:0]            orb_data,
    input  wire [63:0]             timestamp,
    input  wire                    anchor_valid,
    output reg                     anchor_ready,

    output reg  [HASH_WIDTH-1:0]   block_hash,
    output reg                     block_valid
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            anchor_ready <= 1'b1;
            block_valid <= 1'b0;
            block_hash <= 0;
        end else if (anchor_valid && anchor_ready) begin
            // Simplified hashing logic for RTL spec
            block_hash <= orb_data ^ {192'b0, timestamp};
            block_valid <= 1'b1;
            anchor_ready <= 1'b0;
        end else if (block_valid) begin
            block_valid <= 1'b0;
            anchor_ready <= 1'b1;
        end
    end

endmodule
