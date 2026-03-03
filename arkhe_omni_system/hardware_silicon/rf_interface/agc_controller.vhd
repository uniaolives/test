-- agc_controller.vhd
-- Ajusta o ganho do VVA baseado na potência média do sinal

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity agc_controller is
    port (
        clk         : in  std_logic;  -- 100 MHz (clk_rf)
        rst         : in  std_logic;
        i_sample    : in  signed(11 downto 0);  -- amostra I do ADC
        q_sample    : in  signed(11 downto 0);  -- amostra Q
        gain_setting : out std_logic_vector(7 downto 0)  -- para DAC do VVA
    );
end entity;

architecture rtl of agc_controller is
    signal power_sum : unsigned(31 downto 0);
    signal sample_count : unsigned(15 downto 0);
    constant TARGET_POWER : unsigned(31 downto 0) := x"1000000";  -- alvo
begin
    process(clk, rst)
        variable power : unsigned(23 downto 0);
    begin
        if rst = '1' then
            power_sum <= (others => '0');
            sample_count <= (others => '0');
            gain_setting <= x"80";  -- ganho médio inicial
        elsif rising_edge(clk) then
            -- Acumula potência (I² + Q²)
            power := unsigned(abs(i_sample) * abs(i_sample) + abs(q_sample) * abs(q_sample));
            power_sum <= power_sum + power;
            sample_count <= sample_count + 1;

            -- A cada 1024 amostras (~10 µs), atualiza ganho
            if sample_count = 1024 then
                if power_sum > TARGET_POWER then
                    gain_setting <= std_logic_vector(unsigned(gain_setting) - 1);
                elsif power_sum < TARGET_POWER - TARGET_POWER/10 then
                    gain_setting <= std_logic_vector(unsigned(gain_setting) + 1);
                end if;
                power_sum <= (others => '0');
                sample_count <= (others => '0');
            end if;
        end if;
    end process;
end architecture;
