# instaweb_node_v1.0.xdc
# Timing constraints para Xilinx Kria KR260

# Clock principal (SyncE recuperado)
create_clock -period 5.000 -name clk_200m [get_ports clk_200m_p]
create_clock -period 50.000 -name clk_20m [get_ports clk_20m_p]

# IO constraints para sinais ópticos (alta velocidade)
set_input_delay -clock clk_200m -max 1.500 [get_ports optical_rx_p]
set_input_delay -clock clk_200m -min 0.500 [get_ports optical_rx_p]
set_output_delay -clock clk_200m -max 1.500 [get_ports optical_tx_p]
set_output_delay -clock clk_200m -min 0.500 [get_ports optical_tx_p]

# False paths (reset assíncrono)
set_false_path -from [get_ports rst_n]

# Multi-cycle paths (lógica hiperbólica não-crítica)
set_multicycle_path -setup 2 -from [get_cells hyperbolic_calculator/*] -to [get_cells routing_table/*]

# Placement constraints (minimizar skew)
set_property LOC GTHE4_CHANNEL_X0Y0 [get_cells optical_transceiver/gth_inst]
set_property LOC PLLE3_ADV_X0Y0 [get_cells clocking/pll_inst]
