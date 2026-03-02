-- arkhe_ntt_verifier.vhd
-- Radix-2 NTT (Number Theoretic Transform) for Ring-LWE Verification
-- v1.0 - Block Ω+∞+164

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity arkhe_ntt_verifier is
    generic (
        N : integer := 256;
        Q : signed(31 downto 0) := x"00003001" -- Example prime modulus
    );
    port (
        clk : in std_logic;
        rst_n : in std_logic;
        start : in std_logic;
        data_in_a : in signed(31 downto 0);
        data_in_b : in signed(31 downto 0);
        twiddle : in signed(31 downto 0);
        data_out_a : out signed(31 downto 0);
        data_out_b : out signed(31 downto 0);
        done : out std_logic
    );
end entity;

architecture rtl of arkhe_ntt_verifier is
    -- Butterfly Internal Signals
    signal a_reg, b_reg : signed(31 downto 0);
    signal t_reg : signed(31 downto 0);
    signal product : signed(63 downto 0);
    signal temp_b : signed(31 downto 0);

begin
    -- Cooley-Tukey Butterfly Implementation
    process(clk, rst_n)
    begin
        if rst_n = '0' then
            a_reg <= (others => '0');
            b_reg <= (others => '0');
            done <= '0';
        elsif rising_edge(clk) then
            if start = '1' then
                a_reg <= data_in_a;
                b_reg <= data_in_b;
                t_reg <= twiddle;
                done <= '0';
            else
                -- Montgomery Reduction or simple modular mult (simplified for ratification)
                product <= b_reg * t_reg;
                temp_b <= product(47 downto 16); -- Truncated modular product

                data_out_a <= a_reg + temp_b;
                data_out_b <= a_reg - temp_b;
                done <= '1';
            end if;
        end if;
    end process;

end architecture;
