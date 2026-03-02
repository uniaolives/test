-- arkhe_kalman_adaptive.vhd
-- Adaptive Kalman Filter for Doppler Tracking
-- v1.0 - Block Ω+∞+164

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity arkhe_kalman_adaptive is
    generic (
        DT : signed(31 downto 0) := x"00000001"; -- 10ns in fixed-point
        Q_NOMINAL : unsigned(31 downto 0) := x"00000100";
        R_MEAS : unsigned(31 downto 0) := x"00000400";
        MANEUVER_THRESHOLD : unsigned(31 downto 0) := x"00001000"
    );
    port (
        clk : in std_logic;
        rst_n : in std_logic;
        meas_phase : in signed(31 downto 0);
        pred_phase : out signed(31 downto 0);
        coherence : out unsigned(31 downto 0)
    );
end entity;

architecture rtl of arkhe_kalman_adaptive is
    -- State vector (3x1, Q16.16 fixed-point)
    signal x_phase : signed(31 downto 0) := (others => '0');
    signal x_freq : signed(31 downto 0) := (others => '0');
    signal x_accel : signed(31 downto 0) := (others => '0');

    -- Covariance matrix P diagonal
    signal P_11 : unsigned(31 downto 0) := x"00010000";

    -- Innovation
    signal innovation : signed(31 downto 0);
    signal innovation_abs : unsigned(31 downto 0);
    signal Q_adaptive : unsigned(31 downto 0);
    signal Kalman_gain : unsigned(31 downto 0);

begin
    process(clk, rst_n)
    begin
        if rst_n = '0' then
            x_phase <= (others => '0');
            x_freq <= (others => '0');
            P_11 <= x"00010000";
        elsif rising_edge(clk) then
            -- Prediction step
            x_phase <= x_phase + (x_freq * DT);
            x_freq <= x_freq + (x_accel * DT);

            -- Measurement update
            innovation <= meas_phase - x_phase;
            innovation_abs <= unsigned(abs(innovation));

            -- Adapt Q if large innovation
            if innovation_abs > MANEUVER_THRESHOLD then
                Q_adaptive <= Q_NOMINAL + (innovation_abs / 1024);
            else
                Q_adaptive <= Q_NOMINAL;
            end if;

            -- Kalman gain (simplified)
            Kalman_gain <= P_11 / (P_11 + R_MEAS);

            -- State correction
            x_phase <= x_phase + signed(Kalman_gain * unsigned(innovation) / 65536);

            -- Covariance update
            P_11 <= (65536 - Kalman_gain) * P_11 / 65536 + Q_adaptive;
        end if;
    end process;

    pred_phase <= x_phase;
    coherence <= x"10000" - unsigned(innovation_abs); -- Simplified coherence

end architecture;
