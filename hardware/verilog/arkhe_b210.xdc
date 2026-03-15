# B210 Arkhe(n) Constraints
# Target: Spartan-6 XC6SLX75-3FGG484

# Master clock from AD9361 reference (61.44 MHz)
create_clock -period 16.276 -name radio_clk [get_ports clk]

# Phase-critical paths
# Ensure phase registers to sector selection logic is prioritized
set_max_delay -from [get_cells phase_raw_reg*] -to [get_cells best_sector_reg*] 15.0
set_false_path -from [get_cells rst_n] -to [all_registers]

# BRAM initialization constraint
set_property INIT_FILE sacks_lut.hex [get_cells lut_inst/mem]

# Don't touch the golden ratio constant during optimization
set_property DONT_TOUCH true [get_cells *golden_step*]

# IO Placement (Example for USRP B210 expansion headers if needed)
# set_property PACKAGE_PIN AB1 [get_ports {debug_phase_state[0]}]
