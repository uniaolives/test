module quaternion_unit (
    input wire clk,
    input wire rst_n,

    input wire [127:0] q_in,      // [w, x, y, z] (32-bit each)
    input wire [31:0] rotation_angle,
    output reg [127:0] q_out
);

    wire [31:0] w1 = q_in[127:96];
    wire [31:0] x1 = q_in[95:64];
    wire [31:0] y1 = q_in[63:32];
    wire [31:0] z1 = q_in[31:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q_out <= 0;
        end else begin
            // Möbius Twist Logic:
            // If angle > PI, we invert the quaternion (switch "side" of the strip)
            if (rotation_angle > 32'h3243F6A8) begin // > PI in fixed point (16.16)
                q_out[127:96] <= -w1; // Invert scalar
                q_out[95:64]  <= x1;
                q_out[63:32]  <= y1;
                q_out[31:0]   <= -z1; // Invert temporal component
            end else begin
                q_out <= q_in;
            end
        end
    end

endmodule
