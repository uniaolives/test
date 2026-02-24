-- arkhe_topological_node_top.vhd
-- Top-Level Integration for Arkhe Protocol v1.0.0 Hardware
-- v1.1 - Block Ω+∞+178 (Structural Restoration & Port Alignment)
-- v1.0 - Block Ω+∞+164

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity arkhe_topological_node_top is
    port (
        clk_rf : in std_logic;      -- 100 MHz
        clk_dsp : in std_logic;     -- 200 MHz
        rst_n : in std_logic;

        -- RF Front-end (I/Q from ADC)
        i_data : in signed(15 downto 0);
        q_data : in signed(15 downto 0);

        -- System Status
        coherence_global : out unsigned(31 downto 0);
        security_valid : out std_logic
    );
end entity;

architecture rtl of arkhe_topological_node_top is
    -- Signal declarations
    signal phase_extracted : signed(31 downto 0);
    signal cordic_valid : std_logic;

    signal filtered_phase : signed(31 downto 0);
    signal local_coherence : unsigned(31 downto 0);

    signal yb_valid : std_logic;
    signal yb_done : std_logic;

    signal anneal_active : std_logic;
    signal reset_accums : std_logic;
    -- Signal declarations (simplified)
    signal phase_extracted : signed(31 downto 0);
    signal filtered_phase : signed(31 downto 0);
    signal local_coherence : unsigned(31 downto 0);
    signal yb_valid : std_logic;
    signal anneal_active : std_logic;

begin
    -- 1. Phase Extraction (CORDIC)
    cordic_inst: entity work.arkhe_cordic_phase
        port map (
            clk => clk_rf,
            rst_n => rst_n,
            i_in => i_data,
            q_in => q_data,
            phase_out => phase_extracted,
            valid => cordic_valid
        );

    -- 2. Adaptive Filtering (Kalman)
    kalman_inst: entity work.arkhe_kalman_adaptive
        port map (
            clk => clk_dsp,
            rst_n => rst_n,
            meas_phase => phase_extracted,
            pred_phase => filtered_phase,
            coherence => local_coherence
        );

    -- 3. Consensus Verification (Yang-Baxter)
    yb_inst: entity work.arkhe_yb_accelerator
        port map (
            clk => clk_dsp,
            rst_n => rst_n,
            start => cordic_valid,
            r_re => i_data(15) & i_data(15 downto 0) & '0', -- Dummy mapping
            r_im => q_data(15) & q_data(15 downto 0) & '0', -- Dummy mapping
            yb_valid => yb_valid,
            done => yb_done
        );

    -- 4. Thermodynamic Regulation (Annealing)
    anneal_inst: entity work.arkhe_annealing_fsm
        port map (
            clk => clk_dsp,
            rst_n => rst_n,
            coherence_in => local_coherence,
            trigger_anneal => '0',
            annealing_active => anneal_active,
            reset_accumulators => reset_accums
        );
        port map (clk => clk_rf, rst_n => rst_n, i_in => i_data, q_in => q_data, phase_out => phase_extracted);

    -- 2. Adaptive Filtering (Kalman)
    kalman_inst: entity work.arkhe_kalman_adaptive
        port map (clk => clk_dsp, rst_n => rst_n, meas_phase => phase_extracted, pred_phase => filtered_phase, coherence => local_coherence);

    -- 3. Consensus Verification (Yang-Baxter)
    yb_inst: entity work.arkhe_yb_accelerator
        port map (clk => clk_dsp, rst_n => rst_n, yb_valid => yb_valid);

    -- 4. Thermodynamic Regulation (Annealing)
    anneal_inst: entity work.arkhe_annealing_fsm
        port map (clk => clk_dsp, rst_n => rst_n, coherence_in => local_coherence, trigger_anneal => '0', annealing_active => anneal_active);

    -- Final Outputs
    coherence_global <= local_coherence;
    security_valid <= yb_valid;

end architecture;
