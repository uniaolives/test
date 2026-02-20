-- arkhe_rf_interface.vhd
-- Controle de transceptor S-Band e Loop de Recuperação de Portadora (PLL)
-- Projetado para mitigar Doppler LEO (~±50 kHz @ 2.2 GHz)

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity ArkheRfInterface is
    generic (
        IQ_WIDTH    : integer := 16;
        PLL_ALPHA   : integer := 10; -- Ganho proporcional (escalonado)
        PLL_BETA    : integer := 2   -- Ganho integral (escalonado)
    );
    port (
        clk         : in  std_logic; -- clk_rf (100 MHz)
        rst         : in  std_logic;
        -- Interface ADC (I/Q)
        adc_i       : in  signed(IQ_WIDTH-1 downto 0);
        adc_q       : in  signed(IQ_WIDTH-1 downto 0);
        -- Controle de Transceptor (SPI/I2C para Sintetizador)
        tune_freq   : out std_logic_vector(31 downto 0);
        tune_valid  : out std_logic;
        -- AGC Feedback
        agc_gain_ctrl : out std_logic_vector(7 downto 0);
        -- Saída Sincronizada para o CORDIC
        sync_i      : out signed(IQ_WIDTH-1 downto 0);
        sync_q      : out signed(IQ_WIDTH-1 downto 0);
        carrier_lock : out std_logic
    );
end entity;

architecture rtl of ArkheRfInterface is
    -- Internal signals for AGC
    signal power_acc : unsigned(IQ_WIDTH*2 downto 0);
    signal current_gain : unsigned(7 downto 0) := x"80";

    -- PLL Internal Signals (NCO)
    signal nco_phase : signed(15 downto 0) := (others => '0');
    signal nco_freq  : signed(15 downto 0) := (others => '0');
    signal phase_err : signed(15 downto 0);

begin

    -- 1. Automatic Gain Control (AGC) Logic
    -- Mantém a amplitude estável para o CORDIC
    process(clk, rst)
    begin
        if rst = '1' then
            current_gain <= x"80";
        elsif rising_edge(clk) then
            -- Detector de energia simplificado
            -- gain_ctrl = target - (i^2 + q^2)
            -- Simplificação: ajuste por aproximação sucessiva ou loop proporcional
            agc_gain_ctrl <= std_logic_vector(current_gain);
        end if;
    end process;

    -- 2. Loop de Recuperação de Portadora (PLL Digital)
    -- Compensação de Doppler em tempo real
    process(clk, rst)
        variable mix_i, mix_q : signed(IQ_WIDTH*2-1 downto 0);
    begin
        if rst = '1' then
            nco_phase <= (others => '0');
            carrier_lock <= '0';
        elsif rising_edge(clk) then
            -- Phase Error Detector (Complex Multiply with NCO feedback)
            -- err = adc * conj(nco_ref)

            -- Loop Filter (Proporcional + Integral)
            -- nco_freq = nco_freq + BETA * phase_err
            -- nco_phase = nco_phase + nco_freq + ALPHA * phase_err

            nco_phase <= nco_phase + nco_freq;

            -- Lock Detection
            if abs(phase_err) < x"0010" then
                carrier_lock <= '1';
            else
                carrier_lock <= '0';
            end if;
        end if;
    end process;

    -- Saídas sincronizadas
    sync_i <= adc_i; -- Placeholder para mixagem real
    sync_q <= adc_q;

end architecture;
