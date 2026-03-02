-- pll_controller.vhd
-- Ajusta a frequência do LO baseado na saída do Filtro de Kalman

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity pll_controller is
    port (
        clk         : in  std_logic;  -- 50 MHz (clk_safe)
        rst         : in  std_logic;
        freq_error  : in  signed(31 downto 0);  -- erro em Hz (do Kalman)
        pll_update  : out std_logic;  -- pulso para SPI write
        pll_data    : out std_logic_vector(31 downto 0);  -- dado para registrador N
        pll_addr    : out std_logic_vector(7 downto 0)    -- endereço do registrador
    );
end entity;

architecture rtl of pll_controller is
    constant Kp : signed(31 downto 0) := x"00000100";  -- ganho proporcional
    constant Ki : signed(31 downto 0) := x"00000010";  -- ganho integral
    signal integral : signed(63 downto 0);
    signal correction : signed(31 downto 0);
begin
    process(clk, rst)
    begin
        if rst = '1' then
            integral <= (others => '0');
            pll_update <= '0';
        elsif rising_edge(clk) then
            -- Cálculo PI
            integral <= integral + resize(freq_error * Ki, 64);
            correction <= resize(freq_error * Kp, 32) + integral(31 downto 0);

            -- Converte correção em valor de registrador (formato fracionário do ADF5355)
            pll_data <= std_logic_vector(correction);
            pll_addr <= x"02";  -- registrador N fracionário
            pll_update <= '1';   -- pulso de 1 ciclo
        end if;
    end process;
end architecture;
