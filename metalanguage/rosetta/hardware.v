module latent_multiplier(
    input clk,
    input [31:0] agent_vector,
    output reg [31:0] synergy_out
);
    always @(posedge clk) begin
        synergy_out <= agent_vector * 32'h00000168; // Razão Áurea aproximada
    end
endmodule
