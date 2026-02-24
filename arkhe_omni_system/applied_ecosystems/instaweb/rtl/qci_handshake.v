// qci_handshake.v
// Interface entre instaweb (classico) e qhttp (quantico)
// ASI-Omega Quantum-Classical Interface Handshake

module qci_handshake (
    input wire clk,              // 100MHz
    input wire rst_n,

    // Interface classica (instaweb)
    input wire [7:0] classical_data,
    input wire classical_valid,
    output reg classical_ready,

    // Interface quantica (qhttp)
    output reg [3:0] quantum_correction,
    output reg quantum_valid,
    input wire quantum_ready,

    // Status de sincronizacao
    output reg sync_acquired,
    output reg [15:0] buffer_occupancy
);

// Estados do handshake qhttp
typedef enum {IDLE, WAIT_EPR, CORRELATION, APPLY} qstate_t;
qstate_t state;

// Buffer para mensagens classicas (conforme simulacao QCI)
reg [7:0] classical_buffer [0:255];
reg [7:0] wr_ptr, rd_ptr;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        wr_ptr <= 0;
        rd_ptr <= 0;
        sync_acquired <= 0;
        quantum_valid <= 0;
        quantum_correction <= 0;
    end else begin
        buffer_occupancy <= wr_ptr - rd_ptr;

        case (state)
            IDLE: begin
                sync_acquired <= 0;
                if (classical_valid && classical_ready) begin
                    // Fase 1: recebe medicao de Bell via instaweb
                    classical_buffer[wr_ptr] <= classical_data;
                    wr_ptr <= wr_ptr + 1;

                    // Dispara handshake quantico
                    state <= WAIT_EPR;
                end
            end

            WAIT_EPR: begin
                // Fase 2: aguarda confirmacao de entrelacamento via qhttp
                if (quantum_ready) begin
                    quantum_valid <= 1;
                    quantum_correction <= classical_buffer[rd_ptr][3:0]; // Pauli X/Z
                    state <= CORRELATION;
                end
            end

            CORRELATION: begin
                // Espera o sinal ready do modulo quântico ser desativado ou re-ativado
                if (!quantum_ready) begin
                    quantum_valid <= 0;
                    rd_ptr <= rd_ptr + 1;
                    sync_acquired <= 1;
                    state <= APPLY;
                end
            end

            APPLY: begin
                // Fase 3: aplica correcao
                sync_acquired <= 0;
                if (rd_ptr == wr_ptr) begin
                    state <= IDLE;
                end else begin
                    state <= WAIT_EPR;  // proximo pacote no buffer
                end
            end

            default: state <= IDLE;
        endcase
    end
end

// Controle de fluxo: ready quando buffer tem espaco
assign classical_ready = (wr_ptr - rd_ptr) < 256;

endmodule
