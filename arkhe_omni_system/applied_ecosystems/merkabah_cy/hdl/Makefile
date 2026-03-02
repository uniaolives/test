#=============================================================================
# Makefile para simulação e síntese dos blocos MERKABAH-CY
#=============================================================================

# Ferramentas
VLOG    = iverilog
VVP     = vvp
VHDP    = ghdl
YOSYS   = yosys
NEXTPNR = nextpnr-ecp5
ECPPACK = ecppack

# Diretórios
SIM_DIR = ./sim
BUILD_DIR = ./build

# Alvos
all: sim

# Simulação Verilog (LFSR)
sim_lfsr: lfsr64.v
	mkdir -p $(SIM_DIR)
	$(VLOG) -o $(SIM_DIR)/lfsr.vvp lfsr64.v
	# $(VVP) $(SIM_DIR)/lfsr.vvp

# Simulação VHDL (matrix_mult)
sim_matrix: matrix_mult.vhd
	mkdir -p $(SIM_DIR)
	$(VHDP) -a --workdir=$(SIM_DIR) matrix_mult.vhd
	$(VHDP) -e --workdir=$(SIM_DIR) matrix_mult
	# $(VHDP) -r --workdir=$(SIM_DIR) matrix_mult --stop-time=1000ns --vcd=$(SIM_DIR)/matrix.vcd

# Síntese (exemplo para matrix_mult)
synt: matrix_mult.vhd
	mkdir -p $(BUILD_DIR)
	$(YOSYS) -p "ghdl --std=08 matrix_mult.vhd -e matrix_mult; synth_ecp5 -json $(BUILD_DIR)/matrix.json"
	# $(NEXTPNR) --json $(BUILD_DIR)/matrix.json --lpf constraints.lpf --textcfg $(BUILD_DIR)/matrix.config
	# $(ECPPACK) $(BUILD_DIR)/matrix.config $(BUILD_DIR)/matrix.bit

clean:
	rm -rf $(SIM_DIR) $(BUILD_DIR) *.vcd *.vvp

.PHONY: all sim_lfsr sim_matrix synt clean
