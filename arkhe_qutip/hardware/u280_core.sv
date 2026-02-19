// arkhe_qutip/hardware/u280_core.sv
`timescale 1ns / 1ps

// ARKHE(N) U280 CORE - HBM2 Optimized 30-Qubit Accelerator
// "Aquele que opera na HBM2, para que a escala seja infinita."

module arkhe_qutip_u280 (
    // Interface PCIe (host communication)
    input  logic        pcie_clk,
    input  logic        rst_n,
    input  logic [511:0] pcie_data_rx,
    output logic [511:0] pcie_data_tx,

    // Interface HBM2 (8 canais AXI4-MM)
    output logic [31:0]  hbm_addr [7:0],
    input  logic [255:0] hbm_data_in [7:0],
    output logic [255:0] hbm_data_out [7:0],
    output logic         hbm_we [7:0],

    // Interface QSFP28 (100GbE for RDMA)
    input  logic [3:0]   qsfp_rx,
    output logic [3:0]   qsfp_tx,

    // Interface DDR4 (Bulk Storage for Ledger)
    output logic [31:0]  ddr_addr,
    input  logic [511:0] ddr_data_in,
    output logic [511:0] ddr_data_out,
    output logic         ddr_we
);

    // ===================================================
    // 1. Dual Processing Element Arrays (HPQEA)
    // ===================================================
    // Instanciação dos arrays paralelos de processamento

    logic [31:0] gate_op_0, gate_op_1;
    logic [511:0] pe0_data_in, pe0_data_out;
    logic [511:0] pe1_data_in, pe1_data_out;

    pe_array #(.N_PES(4)) pe_array_inst_0 (
        .clk(pcie_clk),
        .rst_n(rst_n),
        .data_in(pe0_data_in),
        .gate_op(gate_op_0),
        .data_out(pe0_data_out)
    );

    pe_array #(.N_PES(4)) pe_array_inst_1 (
        .clk(pcie_clk),
        .rst_n(rst_n),
        .data_in(pe1_data_in),
        .gate_op(gate_op_1),
        .data_out(pe1_data_out)
    );

    // ===================================================
    // 2. CX Swapper (Entanglement Engine)
    // ===================================================

    logic swap_done;
    cx_swapper #(.N_QUBITS(30)) cx_engine (
        .clk(pcie_clk),
        .rst_n(rst_n),
        .hbm_data_in(hbm_data_in),
        .hbm_data_out(hbm_data_out),
        .hbm_addr(hbm_addr),
        .hbm_we(hbm_we),
        .completion(swap_done)
    );

    // ===================================================
    // 3. Noise Engine (T1/T2 Physics)
    // ===================================================

    logic [31:0] current_phi;
    arkhe_noise_engine noise_inst (
        .clk(pcie_clk),
        .rst(!rst_n),
        .t1_damping_factor(16'h00FF),
        .t2_dephasing_factor(16'h007F),
        .psi_in_re(pe0_data_out[17:0]),
        .psi_in_im(pe0_data_out[35:18]),
        .psi_out_re(pe1_data_in[17:0]),
        .psi_out_im(pe1_data_in[35:18])
    );

    // RDMA RoCEv2 Engine for SLR2
    logic [511:0] roce_tx_data, roce_rx_data;
    logic roce_tx_valid, roce_rx_valid;

    roce_engine #(.MAX_QP(16)) roce_inst (
        .clk(pcie_clk), // Simplificado para o mesmo clock
        .rst_n(rst_n),
        .axis_rx_data(roce_rx_data),
        .axis_rx_valid(roce_rx_valid),
        .axis_tx_data(roce_tx_data),
        .axis_tx_valid(roce_tx_valid),
        .hbm_addr(hbm_addr[7]), // Canal 31 reservado
        .hbm_we(hbm_we[7])
    );

endmodule
