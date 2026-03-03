// arkhe_pauli_correction.v
// Pauli Phase Correction Module for QCI Handshake
// v1.0 - Block Ω+∞+180 (Instaweb Production)

`timescale 1ns / 1ps

module arkhe_pauli_correction (
    input  wire        clk,           // 100MHz System Clock
    input  wire        rst_n,         // Active Low Reset

    // Quantum Channel Interface (QCI)
    input  wire        epr_pair_ready, // Entangled pair established
    input  wire        m_bit_arrived,  // Classical bit m arrived from Instaweb
    input  wire        m_bit,          // The classical correction bit
    input  wire        coherence_timer_expired, // Coherence window closed

    // Control / Status
    output reg  [2:0]  qci_state,      // Current FSM state
    output reg         correction_applied,
    output reg         qubit_recycled
);

    // FSM States
    localparam IDLE           = 3'd0;
    localparam WAIT_CLASSICAL = 3'd1;
    localparam TELEPORT_COMPL = 3'd2;
    localparam QUBIT_RECYCLE  = 3'd3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            qci_state <= IDLE;
            correction_applied <= 1'b0;
            qubit_recycled <= 1'b0;
        end else begin
            case (qci_state)
                IDLE: begin
                    correction_applied <= 1'b0;
                    qubit_recycled <= 1'b0;
                    if (epr_pair_ready) begin
                        qci_state <= WAIT_CLASSICAL;
                    end
                end

                WAIT_CLASSICAL: begin
                    if (m_bit_arrived) begin
                        // Apply Pauli Correction:
                        // If m=1, apply X gate (flip phase)
                        // If m=0, I gate (identity)
                        correction_applied <= 1'b1;
                        qci_state <= TELEPORT_COMPL;
                    end else if (coherence_timer_expired) begin
                        qci_state <= QUBIT_RECYCLE;
                    end
                end

                TELEPORT_COMPL: begin
                    qci_state <= IDLE;
                end

                QUBIT_RECYCLE: begin
                    qubit_recycled <= 1'b1;
                    qci_state <= IDLE;
                end

                default: qci_state <= IDLE;
            endcase
        end
    end

endmodule
