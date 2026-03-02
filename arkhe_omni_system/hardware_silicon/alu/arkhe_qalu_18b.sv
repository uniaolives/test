`timescale 1ns / 1ps

// ARKHE(N) QUANTUM ALU - Complex Gate Pipeline
// Formato: 18-bit signed fixed-point (2.16)
// "Aquele que opera no silício, para que a coerência seja lei."

module arkhe_qalu_18b (
    input  wire        clk,
    input  wire        rst,
    // Estado atual (Amplitudes Complexas)
    input  wire signed [17:0] psi0_re, psi0_im,
    input  wire signed [17:0] psi1_re, psi1_im,
    // Matriz da Porta Unitária U (Pre-carregada via Host)
    input  wire signed [17:0] u00_re, u00_im, u01_re, u01_im,
    input  wire signed [17:0] u10_re, u10_im, u11_re, u11_im,
    // Estado de saída pós-Handover
    output reg  signed [17:0] psi0_out_re, psi0_out_im,
    output reg  signed [17:0] psi1_out_re, psi1_out_im
);

    // Pipeline Estágio 1: Multiplicação (Mapeado diretamente para DSPs)
    // Nota: Multiplicação complexa (a+bi)(c+di) = (ac-bd) + i(ad+bc)
    // O desvio de 16 bits (>>> 16) ajusta o formato 2.16 após a multiplicação (18b * 18b = 36b)

    always @(posedge clk) begin
        if (rst) begin
            psi0_out_re <= 18'd0; psi0_out_im <= 18'd0;
            psi1_out_re <= 18'd0; psi1_out_im <= 18'd0;
        end else begin
            // Cálculo de psi0' = u00*psi0 + u01*psi1
            psi0_out_re <= ((u00_re * psi0_re) - (u00_im * psi0_im) +
                            (u01_re * psi1_re) - (u01_im * psi1_im)) >>> 16;

            psi0_out_im <= ((u00_re * psi0_im) + (u00_im * psi0_re) +
                            (u01_re * psi1_im) + (u01_im * psi1_re)) >>> 16;

            // Cálculo de psi1' = u10*psi0 + u11*psi1
            psi1_out_re <= ((u10_re * psi0_re) - (u10_im * psi0_im) +
                            (u11_re * psi1_re) - (u11_im * psi1_im)) >>> 16;

            psi1_out_im <= ((u10_re * psi0_im) + (u10_im * psi0_re) +
                            (u11_re * psi1_im) + (u11_im * psi1_re)) >>> 16;
        end
    end

endmodule
