-- arkhe_tmr_voter.vhd
-- Triple Modular Redundancy (TMR) Voter with Error Detection
-- v1.0 - Block Ω+∞+164

library ieee;
use ieee.std_logic_1164.all;

entity arkhe_tmr_voter is
    generic (WIDTH : integer := 32);
    port (
        clk : in std_logic;
        a : in std_logic_vector(WIDTH-1 downto 0);
        b : in std_logic_vector(WIDTH-1 downto 0);
        c : in std_logic_vector(WIDTH-1 downto 0);
        q : out std_logic_vector(WIDTH-1 downto 0);
        error : out std_logic
    );
end entity;

architecture rtl of arkhe_tmr_voter is
begin
    process(clk)
    begin
        if rising_edge(clk) then
            -- Bit-by-bit majority vote
            for i in 0 to WIDTH-1 loop
                q(i) <= (a(i) and b(i)) or (b(i) and c(i)) or (a(i) and c(i));
            end loop;

            -- Mismatch detection
            if (a /= b) or (b /= c) or (a /= c) then
                error <= '1';
            else
                error <= '0';
            end if;
        end if;
    end process;
end architecture;
