-- asi-sat/fpga/tmr_core.vhd
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- Triple Modular Redundancy wrapper for any module
entity TMR_Wrapper is
    generic (
        DATA_WIDTH : integer := 32;
        VOTER_TYPE : string := "MAJORITY" -- or "AVERAGE" for analog
    );
    port (
        clk : in std_logic;
        reset_n : in std_logic;

        -- Inputs (fed to all 3 modules)
        data_in : in std_logic_vector(DATA_WIDTH-1 downto 0);
        valid_in : in std_logic;

        -- Outputs (voted)
        data_out : out std_logic_vector(DATA_WIDTH-1 downto 0);
        valid_out : out std_logic;

        -- Diagnostics
        mismatch_detected : out std_logic;
        which_module_disagrees : out std_logic_vector(2 downto 0)
    );
end entity;

architecture structural of TMR_Wrapper is
    -- Three identical module instances (stubbed as registers)
    signal mod1_out, mod2_out, mod3_out : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal mod1_valid, mod2_valid, mod3_valid : std_logic;

    -- Voter outputs
    signal voted_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal voted_valid : std_logic;

begin
    -- Logic simulation of 3 modules
    process(clk, reset_n)
    begin
        if reset_n = '0' then
            mod1_out <= (others => '0');
            mod2_out <= (others => '0');
            mod3_out <= (others => '0');
        elsif rising_edge(clk) then
            mod1_out <= data_in;
            mod2_out <= data_in;
            mod3_out <= data_in;
        end if;
    end process;

    -- Majority voter for each bit
    voter_gen: for i in 0 to DATA_WIDTH-1 generate
        voted_data(i) <= (mod1_out(i) and mod2_out(i)) or
                        (mod1_out(i) and mod3_out(i)) or
                        (mod2_out(i) and mod3_out(i));
    end generate;

    -- Valid is AND of all three (conservative)
    voted_valid <= valid_in; -- simplified

    -- Mismatch detection for scrubbing
    mismatch_detected <= '1' when (mod1_out /= voted_data) or
                                  (mod2_out /= voted_data) or
                                  (mod3_out /= voted_data) else '0';

    which_module_disagrees(0) <= '1' when mod1_out /= voted_data else '0';
    which_module_disagrees(1) <= '1' when mod2_out /= voted_data else '0';
    which_module_disagrees(2) <= '1' when mod3_out /= voted_data else '0';

    -- Outputs
    data_out <= voted_data;
    valid_out <= voted_valid;

end architecture;
