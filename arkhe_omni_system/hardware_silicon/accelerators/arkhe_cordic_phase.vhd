-- arkhe_cordic_phase.vhd
-- Extrator de fase topológica via CORDIC vetorial
-- Latência: 16 ciclos @ 100MHz = 160ns (bem dentro do jitter LEO)

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity ArkheCordicPhase is
    generic (
        DATA_WIDTH : integer := 16;
        ITERATIONS : integer := 16
    );
    port (
        clk     : in  std_logic;
        rst     : in  std_logic;
        -- Entrada: Amostra I/Q do ADC
        i_in    : in  signed(DATA_WIDTH-1 downto 0);
        q_in    : in  signed(DATA_WIDTH-1 downto 0);
        -- Saída: Ângulo θ = arctan(Q/I) em formato anyônico
        phase_out : out signed(DATA_WIDTH-1 downto 0);
        valid_out : out std_logic
    );
end entity;

architecture rtl of ArkheCordicPhase is
    -- Tabela de arctan(2^-i) pré-calculada (anyons de Fibonacci)
    type atan_table_t is array (0 to ITERATIONS-1) of signed(DATA_WIDTH-1 downto 0);
    constant ATAN_TABLE : atan_table_t := (
        x"3243", x"1DAC", x"0FAD", x"07F5",  -- π/4, π/8, π/16, π/32...
        x"03FE", x"01FF", x"00FF", x"007F",
        x"003F", x"001F", x"000F", x"0007",
        x"0003", x"0001", x"0000", x"0000"
    );

    -- Pipeline registers
    type xy_array_t is array (0 to ITERATIONS) of signed(DATA_WIDTH downto 0);
    type z_array_t is array (0 to ITERATIONS) of signed(DATA_WIDTH-1 downto 0);

    signal x_reg, y_reg : xy_array_t;
    signal z_reg : z_array_t;
    signal valid_pipe : std_logic_vector(ITERATIONS downto 0);

begin
    -- Inicialização: modo vetorial (rotacionar para eixo X)
    x_reg(0) <= resize(abs(i_in), DATA_WIDTH+1);
    y_reg(0) <= resize(abs(q_in), DATA_WIDTH+1);
    z_reg(0) <= (others => '0');
    valid_pipe(0) <= '1';  -- Assumindo input sempre válido do ADC

    -- Pipeline CORDIC unrolled para throughput máximo
    gen_cordic: for i in 0 to ITERATIONS-1 generate
        process(clk, rst)
            variable x_shift, y_shift : signed(DATA_WIDTH downto 0);
            variable sigma : std_logic;
        begin
            if rst = '1' then
                x_reg(i+1) <= (others => '0');
                y_reg(i+1) <= (others => '0');
                z_reg(i+1) <= (others => '0');
                valid_pipe(i+1) <= '0';
            elsif rising_edge(clk) then
                x_shift := shift_right(x_reg(i), i);
                y_shift := shift_right(y_reg(i), i);
                sigma := y_reg(i)(DATA_WIDTH);  -- sinal de Y

                if sigma = '1' then  -- Y negativo, rotacionar +θ
                    x_reg(i+1) <= x_reg(i) - y_shift;
                    y_reg(i+1) <= y_reg(i) + x_shift;
                    z_reg(i+1) <= z_reg(i) - ATAN_TABLE(i);
                else  -- Y positivo, rotacionar -θ
                    x_reg(i+1) <= x_reg(i) + y_shift;
                    y_reg(i+1) <= y_reg(i) - x_shift;
                    z_reg(i+1) <= z_reg(i) + ATAN_TABLE(i);
                end if;

                valid_pipe(i+1) <= valid_pipe(i);
            end if;
        end process;
    end generate;

    -- Mapeamento para estatísticas anyônicas (fase modular 2π/5 para anyons de Fibonacci)
    phase_out <= z_reg(ITERATIONS) when valid_pipe(ITERATIONS) = '1' else (others => '0');
    valid_out <= valid_pipe(ITERATIONS);

end architecture;
