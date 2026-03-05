// hardware/verilog/cluster_upp.v
// Hardware Quorum Sensing

module cluster_upp #(
    parameter N_UNITS = 64,
    parameter W_BITS = 16
)(
    input wire clk,
    input wire rst_n,
    input wire [N_UNITS-1:0] bio_signal,
    output reg quorum_detected
);
    reg [7:0] active_count;
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_count <= 0;
            quorum_detected <= 0;
        end else begin
            active_count = 0;
            for (i = 0; i < N_UNITS; i = i + 1) begin
                if (bio_signal[i]) active_count = active_count + 1;
            end
            quorum_detected <= (active_count > N_UNITS/2);
        end
    end
endmodule
