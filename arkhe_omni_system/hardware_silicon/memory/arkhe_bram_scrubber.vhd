-- arkhe_bram_scrubber.vhd
-- Periodic BRAM Scrubbing to Correct SEUs
-- v1.0 - Block Ω+∞+164

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity arkhe_bram_scrubber is
    port (
        clk : in std_logic;
        rst_n : in std_logic;
        bram_addr : out std_logic_vector(9 downto 0);
        bram_din : out std_logic_vector(31 downto 0);
        bram_dout : in std_logic_vector(31 downto 0);
        bram_we : out std_logic
    );
end entity;

architecture rtl of arkhe_bram_scrubber is
    signal scrub_addr : unsigned(9 downto 0) := (others => '0');
    signal scrub_timer : unsigned(23 downto 0) := (others => '0');
    constant SCRUB_INTERVAL : unsigned(23 downto 0) := x"4C4B40"; -- ~0.1s @ 50MHz
begin
    process(clk, rst_n)
    begin
        if rst_n = '0' then
            scrub_addr <= (others => '0');
            scrub_timer <= (others => '0');
            bram_we <= '0';
        elsif rising_edge(clk) then
            scrub_timer <= scrub_timer + 1;

            if scrub_timer >= SCRUB_INTERVAL then
                bram_we <= '1';
                bram_din <= bram_dout; -- ECC hardware corrects on read, write back clean
                bram_addr <= std_logic_vector(scrub_addr);
                scrub_addr <= scrub_addr + 1;
                scrub_timer <= (others => '0');
            else
                bram_we <= '0';
            end if;
        end if;
    end process;
end architecture;
