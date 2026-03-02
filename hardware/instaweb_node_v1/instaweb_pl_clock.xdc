# instaweb_pl_clock.xdc
# Clock primario: 25MHz diferencial do PL (KR260)
# Referencia: AMD Kria KR260 Support

set_property PACKAGE_PIN C3 [get_ports "clk_25mhz_p"]
set_property IOSTANDARD LVDS [get_ports "clk_25mhz_p"]
set_property PACKAGE_PIN L3 [get_ports "clk_25mhz_n"]
set_property IOSTANDARD LVDS [get_ports "clk_25mhz_n"]

create_clock -period 40.000 -name pl_clk [get_ports "clk_25mhz_p"]

# I/O para transceptor OWC (Adaptado de GPIO mapping)
set_property PACKAGE_PIN H12 [get_ports "owc_tx_data[0]"]
set_property IOSTANDARD LVCMOS33 [get_ports "owc_tx_data[0]"]
set_property PACKAGE_PIN E10 [get_ports "owc_tx_data[1]"]
set_property IOSTANDARD LVCMOS33 [get_ports "owc_tx_data[1]"]
set_property PACKAGE_PIN D10 [get_ports "owc_tx_data[2]"]
set_property IOSTANDARD LVCMOS33 [get_ports "owc_tx_data[2]"]
set_property PACKAGE_PIN C11 [get_ports "owc_tx_data[3]"]
set_property IOSTANDARD LVCMOS33 [get_ports "owc_tx_data[3]"]
set_property PACKAGE_PIN B11 [get_ports "owc_tx_data[4]"]
set_property IOSTANDARD LVCMOS33 [get_ports "owc_tx_data[4]"]
set_property PACKAGE_PIN A12 [get_ports "owc_tx_data[5]"]
set_property IOSTANDARD LVCMOS33 [get_ports "owc_tx_data[5]"]
set_property PACKAGE_PIN E12 [get_ports "owc_tx_data[6]"]
set_property IOSTANDARD LVCMOS33 [get_ports "owc_tx_data[6]"]
set_property PACKAGE_PIN F12 [get_ports "owc_tx_data[7]"]
set_property IOSTANDARD LVCMOS33 [get_ports "owc_tx_data[7]"]

# Pino para LED indicador de sincronia SyncE
set_property PACKAGE_PIN B10 [get_ports "sync_led"]
set_property IOSTANDARD LVCMOS33 [get_ports "sync_led"]
