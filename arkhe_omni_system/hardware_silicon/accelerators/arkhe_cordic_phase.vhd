-- arkhe_cordic_phase.vhd
-- Phase Extraction from I/Q using Structural CORDIC
-- v1.1 - Block Ω+∞+178 (Structural Restoration)

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity arkhe_cordic_phase is
    generic (
        ITERATIONS : integer := 16
    );
    port (
        clk : in std_logic;
        rst_n : in std_logic;
        i_in : in signed(15 downto 0);
        q_in : in signed(15 downto 0);
        phase_out : out signed(31 downto 0);
        valid : out std_logic
    );
end entity;

architecture structural of arkhe_cordic_phase is
    type angle_array is array (0 to 15) of signed(31 downto 0);
    constant atan_table : angle_array := (
        x"00004000", x"000025C8", x"000013FA", x"00000A22",
        x"00000511", x"00000288", x"00000144", x"000000A2",
        x"00000051", x"00000028", x"00000014", x"0000000A",
        x"00000005", x"00000002", x"00000001", x"00000000"
    );

    signal x, y : signed(15 downto 0);
    signal z : signed(31 downto 0);
    signal count : integer range 0 to ITERATIONS;
begin
    process(clk, rst_n)
        variable x_next, y_next : signed(15 downto 0);
        variable z_next : signed(31 downto 0);
    begin
        if rst_n = '0' then
            x <= (others => '0');
            y <= (others => '0');
            z <= (others => '0');
            count <= 0;
            valid <= '0';
        elsif rising_edge(clk) then
            if count = 0 then
                -- Initial rotation (quadrant adjustment)
                if i_in >= 0 then
                    x <= i_in;
                    y <= q_in;
                    z <= (others => '0');
                else
                    x <= -i_in;
                    y <= -q_in;
                    z <= x"00008000"; -- 180 degrees
                end if;
                count <= 1;
                valid <= '0';
            elsif count <= ITERATIONS then
                -- Iterative rotation
                if y >= 0 then
                    x_next := x + shift_right(y, count-1);
                    y_next := y - shift_right(x, count-1);
                    z_next := z + atan_table(count-1);
                else
                    x_next := x - shift_right(y, count-1);
                    y_next := y + shift_right(x, count-1);
                    z_next := z - atan_table(count-1);
                end if;
                x <= x_next;
                y <= y_next;
                z <= z_next;

                if count = ITERATIONS then
                    phase_out <= z;
                    valid <= '1';
                    count <= 0; -- Reset for next input
                else
                    count <= count + 1;
                end if;
            end if;
        end if;
    end process;
end architecture;
