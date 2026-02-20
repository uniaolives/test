-- arkhe_cdc_fifo.vhd
-- FIFO assíncrona com ponteiros Gray para handover de fase topológica

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity arkhe_cdc_fifo is
    generic (
        DATA_WIDTH : integer := 18;
        ADDR_WIDTH : integer := 4  -- 16 palavras de profundidade
    );
    port (
        -- Domínio de escrita (clk_dsp)
        wclk    : in  std_logic;
        wrst    : in  std_logic;
        wdata   : in  std_logic_vector(DATA_WIDTH-1 downto 0);
        wreq    : in  std_logic;
        wfull   : out std_logic;

        -- Domínio de leitura (clk_safe)
        rclk    : in  std_logic;
        rrst    : in  std_logic;
        rdata   : out std_logic_vector(DATA_WIDTH-1 downto 0);
        rreq    : in  std_logic;
        rempty  : out std_logic
    );
end entity;

architecture rtl of arkhe_cdc_fifo is
    -- Binary to Gray conversion
    function bin_to_gray(bin : unsigned) return unsigned is
    begin
        return bin xor shift_right(bin, 1);
    end function;

    -- Memória (BRAM)
    type mem_t is array (0 to 2**ADDR_WIDTH-1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal mem : mem_t;

    signal wr_ptr_bin, rd_ptr_bin : unsigned(ADDR_WIDTH downto 0) := (others => '0');
    signal wr_ptr_gray, rd_ptr_gray : unsigned(ADDR_WIDTH downto 0) := (others => '0');

    signal wr_ptr_gray_sync1, wr_ptr_gray_sync2 : unsigned(ADDR_WIDTH downto 0) := (others => '0');
    signal rd_ptr_gray_sync1, rd_ptr_gray_sync2 : unsigned(ADDR_WIDTH downto 0) := (others => '0');

begin
    -- Escrita
    process(wclk, wrst)
    begin
        if wrst = '1' then
            wr_ptr_bin <= (others => '0');
            wr_ptr_gray <= (others => '0');
        elsif rising_edge(wclk) then
            if wreq = '1' and wr_ptr_bin(ADDR_WIDTH) = rd_ptr_gray_sync2(ADDR_WIDTH) then -- Not full check simplified
                 -- mem(to_integer(wr_ptr_bin(ADDR_WIDTH-1 downto 0))) <= wdata;
                 -- wr_ptr_bin <= wr_ptr_bin + 1;
            end if;
            -- Full condition: wr_ptr_gray == (rd_ptr_gray_sync2 with MSB inverted)
            -- Simplification for the architectural skeleton provided by the user.
        end if;
    end process;

    -- Sincronizadores e leitura seriam implementados aqui...
    -- Seguindo a estrutura solicitada.

    rdata <= (others => '0');
    rempty <= '1';
    wfull <= '0';

end architecture;
