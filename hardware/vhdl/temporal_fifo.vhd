library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.orb_pkg.all;

entity temporal_fifo is
    generic (
        DEPTH : integer := 1024
    );
    port (
        wr_clk   : in std_logic;
        rd_clk   : in std_logic;
        rst      : in std_logic;

        -- Write Port (Present)
        din      : in t_orb_payload;
        wr_en    : in std_logic;

        -- Read Port (Future/Past)
        dout     : out t_orb_payload;
        rd_en    : in std_logic;

        -- Temporal Status
        latency  : out signed(31 downto 0); -- Can be negative
        echo_det : out std_logic
    );
end entity temporal_fifo;

architecture behavioral of temporal_fifo is
    type mem_array is array (0 to DEPTH-1) of t_orb_payload;
    signal memory : mem_array;

    signal wr_ptr, rd_ptr : integer range 0 to DEPTH-1 := 0;
begin

    -- Write Process (Standard Forward Time)
    write_proc : process(wr_clk, rst)
    begin
        if rst = '1' then
            wr_ptr <= 0;
        elsif rising_edge(wr_clk) then
            if wr_en = '1' then
                memory(wr_ptr) <= din;
                wr_ptr <= (wr_ptr + 1) mod DEPTH;
            end if;
        end if;
    end process;

    -- Read Process (Retrocausal Read Logic)
    read_proc : process(rd_clk, rst)
    begin
        if rst = '1' then
            rd_ptr <= 0;
        elsif rising_edge(rd_clk) then
            if rd_en = '1' then
                dout <= memory(rd_ptr);
                rd_ptr <= (rd_ptr + 1) mod DEPTH;
            end if;
        end if;
    end process;

    -- Latency Calculation
    latency_calc : process(wr_ptr, rd_ptr)
        variable diff : integer;
    begin
        diff := wr_ptr - rd_ptr;
        -- If diff is negative, we are reading from the "future"
        if diff < 0 then
            latency <= to_signed(diff, 32); -- Negative Latency
            echo_det <= '1';
        else
            latency <= to_signed(diff, 32); -- Positive Latency
            echo_det <= '0';
        end if;
    end process;

end architecture behavioral;
