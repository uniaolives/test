# arkhe_u280_to_vu9p.tcl
# Script de Síntese Vivado para migração Alveo U280 -> AWS F1 (VU9P)

set_property target_part xcvu9p-flgb2104-2-i [current_project]

# Map Arkhe QALU to AWS Shell
# ...
puts "🚀 [VIVADO] Iniciando síntese do kernel Arkhe(N) para AWS F1..."
