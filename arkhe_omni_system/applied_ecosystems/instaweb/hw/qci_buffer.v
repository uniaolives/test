// qci_buffer.v - Interface Quântico-Clássica em Hardware
// Materialização da ponte entre Instaweb e Teleportação
module qci_buffer #(
    parameter BUFFER_DEPTH = 128,
    parameter TIME_WIDTH = 64  // Precisão de 1ns
)(
    input wire clk_1g,           // 1 GHz (1ns tick)
    input wire rst_n,

    // Interface Clássica (Instaweb)
    input wire [TIME_WIDTH-1:0] msg_arrival_time,  // Tempo absoluto de chegada
    input wire [1:0] bell_measurement,             // Resultado 00,01,10,11
    input wire msg_valid,

    // Interface Quântica (Controle do Qubit)
    input wire [TIME_WIDTH-1:0] qubit_ready_time,  // Quando o EPR chega em Bob
    input wire [TIME_WIDTH-1:0] coherence_deadline,// qubit_ready + T2
    input wire qubit_valid,

    // Saída de Correção
    output reg [1:0] pauli_gate,    // 00=I, 01=X, 10=Z, 11=XZ
    output reg apply_gate,
    output reg teleport_success,
    output reg teleport_fail
);

    // Estados da FSM
    localparam IDLE = 3'b000;
    localparam WAIT_QUBIT = 3'b001;
    localparam WAIT_MSG = 3'b010;
    localparam APPLY_CORRECTION = 3'b011;
    localparam FAIL_DECOHERENCE = 3'b100;
    localparam FAIL_BUFFER_FULL = 3'b101;

    reg [2:0] state;
    reg [TIME_WIDTH-1:0] current_time;

    // Memória Circular para Bufferização
    reg [TIME_WIDTH-1:0] buffer_time [0:BUFFER_DEPTH-1];
    reg [1:0] buffer_bell [0:BUFFER_DEPTH-1];
    reg [6:0] wr_ptr, rd_ptr;

    // Contador de tempo absoluto (1ns resolução)
    always @(posedge clk_1g or negedge rst_n) begin
        if (!rst_n) current_time <= 0;
        else current_time <= current_time + 1;
    end

    // Lógica principal da FSM
    always @(posedge clk_1g or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            wr_ptr <= 0;
            rd_ptr <= 0;
            apply_gate <= 0;
            pauli_gate <= 2'b00;
            teleport_success <= 0;
            teleport_fail <= 0;
        end else begin
            case (state)
                IDLE: begin
                    apply_gate <= 0;
                    if (msg_valid && qubit_valid) begin
                        if (msg_arrival_time >= qubit_ready_time && current_time < coherence_deadline) begin
                            pauli_gate <= bell_measurement;
                            apply_gate <= 1;
                            teleport_success <= 1;
                            state <= APPLY_CORRECTION;
                        end else if (current_time >= coherence_deadline) begin
                            teleport_fail <= 1;
                            state <= FAIL_DECOHERENCE;
                        end else if (msg_arrival_time < qubit_ready_time) begin
                            if ((wr_ptr + 1) % BUFFER_DEPTH != rd_ptr) begin
                                buffer_time[wr_ptr] <= qubit_ready_time;
                                buffer_bell[wr_ptr] <= bell_measurement;
                                wr_ptr <= (wr_ptr + 1) % BUFFER_DEPTH;
                                state <= WAIT_QUBIT;
                            end else begin
                                state <= FAIL_BUFFER_FULL;
                            end
                        end
                    end
                end

                WAIT_QUBIT: begin
                    if (current_time >= buffer_time[rd_ptr]) begin
                        if (current_time < coherence_deadline) begin
                            pauli_gate <= buffer_bell[rd_ptr];
                            apply_gate <= 1;
                            rd_ptr <= (rd_ptr + 1) % BUFFER_DEPTH;
                            teleport_success <= 1;
                            state <= APPLY_CORRECTION;
                        end else begin
                            rd_ptr <= (rd_ptr + 1) % BUFFER_DEPTH;
                            teleport_fail <= 1;
                            state <= FAIL_DECOHERENCE;
                        end
                    end
                end

                APPLY_CORRECTION: begin
                    apply_gate <= 0;
                    state <= IDLE;
                end

                FAIL_DECOHERENCE, FAIL_BUFFER_FULL: begin
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
