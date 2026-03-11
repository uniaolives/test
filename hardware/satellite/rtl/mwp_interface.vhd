-- mwp_interface.vhd
-- Microtubular Waveguide Processor Interface
-- Reads 8 analog channels from photodetectors and computes correlation

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity mwp_interface is
    Port ( clk        : in  STD_LOGIC;
           rst_n      : in  STD_LOGIC;
           -- 8 analog inputs from photodetectors (12-bit ADC)
           adc_data   : in  STD_LOGIC_VECTOR(11 downto 0);
           adc_valid  : in  STD_LOGIC;
           -- Control
           sample_sel : out STD_LOGIC_VECTOR(2 downto 0);  -- selects channel
           -- Output
           pattern    : out STD_LOGIC_VECTOR(31 downto 0); -- interference result
           pattern_valid : out STD_LOGIC
    );
end mwp_interface;

architecture Behavioral of mwp_interface is
    type adc_array is array (0 to 7) of unsigned(11 downto 0);
    signal samples : adc_array := (others => (others => '0'));
    signal channel : integer range 0 to 7 := 0;
    signal acc     : unsigned(15 downto 0) := (others => '0');
    signal cnt     : integer range 0 to 7 := 0;
    type state_t is (ACQUIRE, COMPUTE);
    signal state : state_t := ACQUIRE;
begin

    process(clk, rst_n)
    begin
        if rst_n = '0' then
            channel <= 0;
            samples <= (others => (others => '0'));
            pattern <= (others => '0');
            pattern_valid <= '0';
            state <= ACQUIRE;
            acc <= (others => '0');
            cnt <= 0;
            sample_sel <= (others => '0');
        elsif rising_edge(clk) then
            case state is
                when ACQUIRE =>
                    sample_sel <= std_logic_vector(to_unsigned(channel, 3));
                    pattern_valid <= '0';
                    if adc_valid = '1' then
                        samples(channel) <= unsigned(adc_data);
                        if channel = 7 then
                            state <= COMPUTE;
                            cnt <= 0;
                        else
                            channel <= channel + 1;
                        end if;
                    end if;

                when COMPUTE =>
                    -- Simple dot product (pattern recognition)
                    -- In real MWP, this would be a more complex correlation
                    acc <= acc + resize(samples(cnt), 16);
                    if cnt = 7 then
                        pattern <= std_logic_vector(resize(acc, 32));
                        pattern_valid <= '1';
                        acc <= (others => '0');
                        state <= ACQUIRE;
                        channel <= 0;
                    else
                        cnt <= cnt + 1;
                    end if;
            end case;
        end if;
    end process;

end Behavioral;
