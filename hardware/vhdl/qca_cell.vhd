library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity qca_cell is
    generic (
        LEVEL : integer range 0 to 1023
    );
    port (
        clk             : in  std_logic;
        rst             : in  std_logic;
        lambda2         : in  std_logic_vector(15 downto 0);  -- 0..1 Q16
        barrier_width   : in  std_logic_vector(63 downto 0);  -- |delta_t|
        confinement_mode: in  std_logic_vector(1 downto 0);   -- 00 INF, 01 FINITE, 10 BARRIER, 11 FREE
        probability     : out std_logic_vector(31 downto 0)   -- Q32
    );
end entity;

architecture behavioral of qca_cell is
begin
    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                probability <= (others => '0');
            else
                -- Simplified probability calculation for RTL specification
                -- In a real implementation, this would use a CORDIC/Taylor series or LUT
                case confinement_mode is
                    when "00" =>  -- INFINITE_WELL
                        probability <= x"FFFFFFFF"; -- 1.0
                    when "01" =>  -- FINITE_WELL
                        probability <= x"E0000000"; -- 0.875
                    when "10" =>  -- BARRIER
                        probability <= x"80000000"; -- 0.5
                    when others =>  -- FREE
                        probability <= x"20000000"; -- 0.125
                end case;
            end if;
        end if;
    end process;
end architecture;
