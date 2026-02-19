// arkhe_qutip/hardware/pe_array.sv
`timescale 1ns / 1ps

// HPQEA - High-Performance Quantum Emulation Element Array
// Processamento paralelo de amplitudes quânticas.

module pe_array #(
    parameter N_PES = 4
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic [511:0] data_in,
    input  logic [31:0]  gate_op,
    output logic [511:0] data_out
);

    // Cada PE (Processing Element) opera em um par de amplitudes complexas
    genvar i;
    generate
        for (i = 0; i < N_PES; i++) begin : pe_loop
            // Mapeia para blocos DSP48E2
            arkhe_qalu_18b alu_inst (
                .clk(clk),
                .rst(!rst_n),
                .psi0_re(data_in[i*128 +: 18]),
                .psi0_im(data_in[i*128+18 +: 18]),
                .psi1_re(data_in[i*128+64 +: 18]),
                .psi1_im(data_in[i*128+82 +: 18]),
                .u00_re(gate_op[0 +: 18]), // Simplificado: gates são passados via bus
                .u00_im(18'd0),
                .u01_re(gate_op[18 +: 18]),
                .u01_im(18'd0),
                .u10_re(18'd0),
                .u10_im(18'd0),
                .u11_re(18'd1 << 16), // Identidade parcial
                .u11_im(18'd0),
                .psi0_out_re(data_out[i*128 +: 18]),
                .psi0_out_im(data_out[i*128+18 +: 18]),
                .psi1_out_re(data_out[i*128+64 +: 18]),
                .psi1_out_im(data_out[i*128+82 +: 18])
            );
        end
    endgenerate

endmodule
