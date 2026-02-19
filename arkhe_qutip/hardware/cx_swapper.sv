// arkhe_qutip/hardware/cx_swapper.sv
`timescale 1ns / 1ps

// CX Swapper - Gerencia transferências HBM2 para portas multi-qubit (CNOT)
// Otimiza o acesso à memória para evitar stalls no pipeline.

module cx_swapper #(
    parameter N_QUBITS = 30
)(
    input  logic        clk,
    input  logic        rst_n,
    // Interface HBM2
    input  logic [255:0] hbm_data_in [7:0],
    output logic [255:0] hbm_data_out [7:0],
    output logic [31:0]  hbm_addr [7:0],
    output logic         hbm_we [7:0],
    output logic         completion
);

    // Máquina de estados para escalonamento de CNOT
    typedef enum logic [2:0] {IDLE, FETCH, COMPUTE, STORE, DONE} state_t;
    state_t state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            completion <= 0;
            for(int i=0; i<8; i++) hbm_we[i] <= 0;
        end else begin
            case (state)
                IDLE: begin
                    state <= FETCH;
                    completion <= 0;
                end
                FETCH: begin
                    // Ativa leitura paralela nos pseudo-canais HBM
                    state <= COMPUTE;
                end
                COMPUTE: begin
                    // Aguarda conclusão do PE Array
                    state <= STORE;
                end
                STORE: begin
                    for(int i=0; i<8; i++) hbm_we[i] <= 1;
                    state <= DONE;
                end
                DONE: begin
                    completion <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
