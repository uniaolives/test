//=============================================================================
// lfsr64.v
// Gerador de números pseudo‑aleatórios de 64 bits (LFSR) para inicialização.
//=============================================================================

module lfsr64 (
    input  wire        clk,
    input  wire        rst,
    input  wire        enable,
    output wire [63:0] random_out
);

    reg [63:0] state;
    wire [63:0] next_state;

    // Tap positions for maximal length LFSR (64-bit, polynomial x^64 + x^63 + x^61 + x^60 + 1)
    // Not maximal but good enough for mock
    assign next_state = {state[62:0],
                         state[63] ^ state[62] ^ state[60] ^ state[59]};

    always @(posedge clk or posedge rst) begin
        if (rst)
            state <= 64'hdeadbeef12345678;  // seed
        else if (enable)
            state <= next_state;
    end

    assign random_out = state;

endmodule
