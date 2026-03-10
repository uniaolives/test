-- qca_cell_casimir.vhd (Updated with White et al. 2026)

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use IEEE.MATH_REAL.ALL;

entity qca_cell_casimir is
    generic (
        H_BAR : real := 1.054571817e-34;
        MU    : real := 9.104425e-31;
        N_QUANTUM : integer range 1 to 7 := 1
    );
    port (
        clk : in std_logic;
        rst : in std_logic;
        delta_t_seconds : in real;
        lambda_2_in     : in real;
        tunnel_probability : out real;
        is_bound : out std_logic
    );
end entity;

architecture casimir of qca_cell_casimir is
begin
    process(clk)
        variable A_eff : real;
    begin
        if rising_edge(clk) then
            if rst = '1' then
                is_bound <= '0';
            else
                A_eff := -real(N_QUANTUM * N_QUANTUM) * 0.156433;
                is_bound <= '1' when (A_eff < 0.0 and lambda_2_in > 0.8) else '0';

                -- T ~ exp(-2 * |delta_t| * (1 - lambda_2) * sqrt(|A|))
                tunnel_probability <= exp(-2.0 * abs(delta_t_seconds) *
                                        (1.0 - lambda_2_in) *
                                        sqrt(abs(A_eff)));
            end if;
        end if;
    end process;
end architecture;
