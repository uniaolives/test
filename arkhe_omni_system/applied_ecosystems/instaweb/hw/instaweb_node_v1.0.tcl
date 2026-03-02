# instaweb_node_v1.0.tcl - Script de síntese Vivado

# Criar projeto
create_project instaweb_node_v1.0 ./instaweb_node_v1.0 -part xck26-sfvc784-2LV-c

# Adicionar sources
add_files [glob ../rtl/*.v ../rtl/*.sv]
add_files ../constraints/instaweb_node_v1.0.xdc

# Configurar estratégia de síntese (performance máxima)
set_property strategy Performance_Explore [get_runs synth_1]
set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]

# Compilar
launch_runs synth_1 -jobs 8
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

# Exportar
write_hw_platform -fixed -include_bit -file instaweb_node_v1.0.xsa
