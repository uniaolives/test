-- arkhe_annealing_fsm.vhd
-- Thermodynamic Annealing State Machine for Coherence Recovery
-- v1.0 - Block Ω+∞+164

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity arkhe_annealing_fsm is
    port (
        clk : in std_logic;
        rst_n : in std_logic;
        coherence_in : in unsigned(31 downto 0);
        trigger_anneal : in std_logic;
        annealing_active : out std_logic;
        reset_accumulators : out std_logic
    );
end entity;

architecture rtl of arkhe_annealing_fsm is
    type state_type is (IDLE, HEATING, COOLING, STABILIZED);
    signal state : state_type := IDLE;
    signal timer : unsigned(31 downto 0) := (others => '0');
    constant PSI_THRESHOLD : unsigned(31 downto 0) := x"0000D8D4"; -- 0.847 in Q16.16
begin
    process(clk, rst_n)
    begin
        if rst_n = '0' then
            state <= IDLE;
            annealing_active <= '0';
            reset_accumulators <= '0';
        elsif rising_edge(clk) then
            case state is
                when IDLE =>
                    if trigger_anneal = '1' or coherence_in < PSI_THRESHOLD then
                        state <= HEATING;
                        annealing_active <= '1';
                        reset_accumulators <= '1';
                    end if;

                when HEATING =>
                    reset_accumulators <= '0';
                    if timer > 1000 then -- 5us heating
                        state <= COOLING;
                        timer <= (others => '0');
                    else
                        timer <= timer + 1;
                    end if;

                when COOLING =>
                    if coherence_in > PSI_THRESHOLD then
                        state <= STABILIZED;
                    end if;

                when STABILIZED =>
                    annealing_active <= '0';
                    state <= IDLE;
            end case;
        end if;
    end process;
end architecture;
