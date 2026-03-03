--=============================================================================
-- matrix_mult.vhd
-- Multiplicador de matriz vetor (sistólico) para o kernel de coerência.
-- Entradas: vetor de features (x) e matriz de interseção (a).
-- Saída: y = A * x
--=============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity matrix_mult is
    generic (
        N : integer := 100;          -- Dimensão da matriz (h11)
        DW : integer := 32            -- Largura dos dados (fixo‑ponto)
    );
    port (
        clk         : in  std_logic;
        rst         : in  std_logic;
        start       : in  std_logic;
        done        : out std_logic;
        -- Entrada do vetor x (interface de memória)
        x_addr      : out std_logic_vector(31 downto 0);
        x_data      : in  std_logic_vector(DW-1 downto 0);
        x_valid     : in  std_logic;
        -- Entrada da matriz A (carregada sequencialmente)
        a_data      : in  std_logic_vector(DW-1 downto 0);
        a_valid     : in  std_logic;
        -- Saída do vetor y
        y_data      : out std_logic_vector(DW-1 downto 0);
        y_valid     : out std_logic
    );
end entity matrix_mult;

architecture systolic of matrix_mult is

    type systolic_row is array (0 to N-1) of signed(DW-1 downto 0);
    type systolic_array is array (0 to N-1) of systolic_row;

    signal mac_vals : systolic_array := (others => (others => (others => '0')));
    signal a_shift   : systolic_row;
    signal x_reg     : signed(DW-1 downto 0);
    signal cnt       : integer range 0 to N*N;

    type state_type is (idle, load_a, load_x, compute, finish);
    signal state : state_type;

begin

    process(clk, rst)
        variable sum : signed(DW-1 downto 0);
    begin
        if rst = '1' then
            state <= idle;
            done <= '0';
            y_valid <= '0';
            cnt <= 0;
        elsif rising_edge(clk) then
            case state is
                when idle =>
                    if start = '1' then
                        state <= load_a;
                        cnt <= 0;
                        done <= '0';
                    end if;

                when load_a =>
                    if a_valid = '1' then
                        -- Alimenta a primeira coluna da matriz
                        a_shift(0) <= signed(a_data);
                        for i in 1 to N-1 loop
                            a_shift(i) <= a_shift(i-1);
                        end loop;
                        cnt <= cnt + 1;
                        if cnt = N*N - 1 then
                            state <= load_x;
                            cnt <= 0;
                        end if;
                    end if;

                when load_x =>
                    if x_valid = '1' then
                        x_reg <= signed(x_data);
                        cnt <= cnt + 1;
                        if cnt = N-1 then
                            state <= compute;
                            cnt <= 0;
                        end if;
                    end if;

                when compute =>
                    -- MAC operação
                    for i in 0 to N-1 loop
                        for j in 0 to N-1 loop
                            sum := mac_vals(i)(j) + a_shift(j) * x_reg;
                            mac_vals(i)(j) <= sum;
                        end loop;
                    end loop;
                    cnt <= cnt + 1;
                    if cnt = N then
                        state <= finish;
                    end if;

                when finish =>
                    -- Saída do vetor resultado (pipeline)
                    for i in 0 to N-1 loop
                        y_data <= std_logic_vector(mac_vals(i)(N-1));
                        y_valid <= '1';
                    end loop;
                    done <= '1';
                    state <= idle;

            end case;
        end if;
    end process;

    -- Atribuição de endereços (para memória externa)
    x_addr <= std_logic_vector(to_unsigned(cnt, 32));

end architecture systolic;
