-- arkhe_braiding_buffer.vhd
-- Memória triplicada (TMR) com Scrubbing automático para o CubeSat Arkhe-1.
-- Garante a integridade da história anyônica sob radiação ionizante.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity arkhe_braiding_buffer is
    generic (
        ADDR_WIDTH : integer := 10; -- 1024 entradas
        DATA_WIDTH : integer := 64  -- 64-bit anyon state
    );
    port (
        clk         : in  std_logic;
        rst         : in  std_logic;

        -- Porta de Escrita (Sistema)
        wr_en       : in  std_logic;
        wr_addr     : in  unsigned(ADDR_WIDTH-1 downto 0);
        wr_data     : in  std_logic_vector(DATA_WIDTH-1 downto 0);

        -- Porta de Leitura (Acelerador YB)
        rd_addr     : in  unsigned(ADDR_WIDTH-1 downto 0);
        rd_data     : out std_logic_vector(DATA_WIDTH-1 downto 0);

        -- Status
        seu_corrected : out std_logic;
        fatal_error   : out std_logic
    );
end entity;

architecture rtl of arkhe_braiding_buffer is

    -- BRAM triplicada
    type bram_t is array (0 to 2**ADDR_WIDTH - 1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal bram_0, bram_1, bram_2 : bram_t;

    -- Scrubbing State Machine
    type scrub_state_t is (IDLE, READ_BLOCK, VOTE, WRITE_BACK);
    signal state : scrub_state_t;
    signal scrub_addr : unsigned(ADDR_WIDTH-1 downto 0);

    -- Registros de leitura do scrubbing
    signal s_data_0, s_data_1, s_data_2 : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal corrected_word : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal correction_needed : std_logic;

begin

    -- Porta de Escrita Síncrona (Escreve nas 3 cópias simultaneamente)
    process(clk)
    begin
        if rising_edge(clk) then
            if wr_en = '1' then
                bram_0(to_integer(wr_addr)) <= wr_data;
                bram_1(to_integer(wr_addr)) <= wr_data;
                bram_2(to_integer(wr_addr)) <= wr_data;
            elsif state = WRITE_BACK and correction_needed = '1' then
                -- Scrubbing escrevendo o valor corrigido
                bram_0(to_integer(scrub_addr)) <= corrected_word;
                bram_1(to_integer(scrub_addr)) <= corrected_word;
                bram_2(to_integer(scrub_addr)) <= corrected_word;
            end if;
        end if;
    end process;

    -- Porta de Leitura com Votação Majoritária (Latência 1 ciclo)
    process(clk)
        variable d0, d1, d2 : std_logic_vector(DATA_WIDTH-1 downto 0);
    begin
        if rising_edge(clk) then
            d0 := bram_0(to_integer(rd_addr));
            d1 := bram_1(to_integer(rd_addr));
            d2 := bram_2(to_integer(rd_addr));

            -- Votador de bit (simplificado para leitura rápida)
            for i in 0 to DATA_WIDTH-1 loop
                rd_data(i) <= (d0(i) and d1(i)) or (d0(i) and d2(i)) or (d1(i) and d2(i));
            end loop;
        end if;
    end process;

    -- Lógica de Scrubbing Automático (Background)
    process(clk, rst)
        variable match_01, match_02, match_12 : boolean;
    begin
        if rst = '1' then
            state <= IDLE;
            scrub_addr <= (others => '0');
            seu_corrected <= '0';
            fatal_error <= '0';
        elsif rising_edge(clk) then
            seu_corrected <= '0';

            case state is
                when IDLE =>
                    state <= READ_BLOCK;

                when READ_BLOCK =>
                    s_data_0 <= bram_0(to_integer(scrub_addr));
                    s_data_1 <= bram_1(to_integer(scrub_addr));
                    s_data_2 <= bram_2(to_integer(scrub_addr));
                    state <= VOTE;

                when VOTE =>
                    match_01 := (s_data_0 = s_data_1);
                    match_02 := (s_data_0 = s_data_2);
                    match_12 := (s_data_1 = s_data_2);

                    if match_01 and match_02 then
                        -- Saudável
                        correction_needed <= '0';
                        state <= IDLE;
                        scrub_addr <= scrub_addr + 1;
                    elsif match_01 then
                        -- Cópia 2 corrompida
                        corrected_word <= s_data_0;
                        correction_needed <= '1';
                        seu_corrected <= '1';
                        state <= WRITE_BACK;
                    elsif match_02 then
                        -- Cópia 1 corrompida
                        corrected_word <= s_data_0;
                        correction_needed <= '1';
                        seu_corrected <= '1';
                        state <= WRITE_BACK;
                    elsif match_12 then
                        -- Cópia 0 corrompida
                        corrected_word <= s_data_1;
                        correction_needed <= '1';
                        seu_corrected <= '1';
                        state <= WRITE_BACK;
                    else
                        -- Falha catastrófica (múltiplos bits em múltiplas cópias)
                        fatal_error <= '1';
                        state <= IDLE;
                    end if;

                when WRITE_BACK =>
                    -- Escrita realizada no processo síncrono acima
                    state <= IDLE;
                    scrub_addr <= scrub_addr + 1;

            end case;
        end if;
    end process;

end architecture;
