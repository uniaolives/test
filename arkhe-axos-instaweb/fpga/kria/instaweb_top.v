// arkhe-axos-instaweb/fpga/kria/instaweb_top.v
module instaweb_top (
    input wire clk,
    input wire rst_n,
    input wire [31:0] data_in,
    output wire [31:0] data_out
);
    // Instaweb FPGA Top Level
    assign data_out = data_in ^ 32'hDEADBEEF;
endmodule
