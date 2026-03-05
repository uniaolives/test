-- hardware/vhdl/upp_core.vhd
-- Primary Processing Unit Core

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity upp_core is
    port (
        clk : in std_logic;
        rst_n : in std_logic;
        energy_avail : in unsigned(15 downto 0);
        directive : out std_logic_vector(1 downto 0) -- 00=A, 01=S, 10=T
    );
end entity;

architecture behavioral of upp_core is
    type directive_t is (ADAPT, STORE, TRANSMIT);
    signal current_dir : directive_t;
    signal vk_ref : unsigned(15 downto 0);
begin
    process(clk, rst_n)
    begin
        if rst_n = '0' then
            vk_ref <= (others => '0');
            current_dir <= ADAPT;
        elsif rising_edge(clk) then
            if energy_avail > vk_ref + 100 then
                current_dir <= ADAPT;
                vk_ref <= vk_ref + 10;
            elsif energy_avail < vk_ref - 100 then
                current_dir <= TRANSMIT;
            else
                current_dir <= STORE;
            end if;
        end if;
    end process;

    directive <= "00" when current_dir = ADAPT else
                 "01" when current_dir = STORE else
                 "10";
end architecture;
