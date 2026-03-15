// arkhe-os/hardware/fpga/sacks_prime_lut.v
module sacks_prime_lut #(
    parameter ADDR_WIDTH = 16,
    parameter DATA_WIDTH = 64
)(
    input wire clka,
    input wire wea,
    input wire [ADDR_WIDTH-1:0] addra,
    input wire [DATA_WIDTH-1:0] dina,
    output reg [DATA_WIDTH-1:0] douta,

    input wire clkb,
    input wire web,
    input wire [ADDR_WIDTH-1:0] addrb,
    input wire [DATA_WIDTH-1:0] dinb,
    output reg [DATA_WIDTH-1:0] doutb
);

    // Initialized with pre-computed Sacks spiral data
    (* ram_style = "block" *)
    reg [DATA_WIDTH-1:0] mem [0:1023];

    initial begin
        $readmemh("sacks_lut.hex", mem);
    end

    always @(posedge clka) begin
        if (wea) mem[addra] <= dina;
        douta <= mem[addra];
    end

    always @(posedge clkb) begin
        if (web) mem[addrb] <= dinb;
        doutb <= mem[addrb];
    end

endmodule
