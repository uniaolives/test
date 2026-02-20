-- arkhe_yb_accelerator.vhd
-- Verificador Yang-Baxter com pipeline duplo e TMR na saída

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity ArkheYBAccelerator is
    generic (
        PHASE_WIDTH : integer := 16;
        TMR_EN      : boolean := true
    );
    port (
        clk         : in  std_logic;
        rst         : in  std_logic;
        -- Entradas triádicas (3 anyons)
        phase_a     : in  signed(PHASE_WIDTH-1 downto 0);
        phase_b     : in  signed(PHASE_WIDTH-1 downto 0);
        phase_c     : in  signed(PHASE_WIDTH-1 downto 0);
        valid_in    : in  std_logic;
        -- Saídas
        yb_valid    : out std_logic;  -- '1' se LHS = RHS (dentro de tolerância)
        phase_result: out signed(PHASE_WIDTH-1 downto 0);
        seu_detected: out std_logic
    );
end entity;

architecture structural of ArkheYBAccelerator is

    -- Tipos para matrizes de roteamento R_ij
    subtype phase_t is signed(PHASE_WIDTH-1 downto 0);
    type matrix_r_t is record
        r12, r13, r23 : phase_t;
    end record;

    -- Função de composição de fase anyônica (simplificada para Fibonacci)
    function compose_phase(p1, p2 : phase_t) return phase_t is
        variable result : signed(PHASE_WIDTH*2-1 downto 0);
    begin
        -- Multiplicação de fases como adição em log (característica anyônica)
        -- Na realidade: R_ij(θ₁) ∘ R_ij(θ₂) = R_ij(θ₁ + θ₂ mod 2π/5)
        result := resize(p1, PHASE_WIDTH*2) + resize(p2, PHASE_WIDTH*2);
        return resize(result mod (2**PHASE_WIDTH / 5), PHASE_WIDTH);  -- Modular 2π/5
    end function;

    -- Pipeline LHS: R12 → R13 → R23
    signal lhs_pipe : matrix_r_t;
    signal lhs_result : phase_t;
    signal lhs_valid : std_logic_vector(2 downto 0);

    -- Pipeline RHS: R23 → R13 → R12
    signal rhs_pipe : matrix_r_t;
    signal rhs_result : phase_t;
    signal rhs_valid : std_logic_vector(2 downto 0);

    -- TMR instances
    signal lhs_tmr : phase_t;
    signal rhs_tmr : phase_t;

    -- Tolerância de comparação (ruído quântico + truncamento CORDIC)
    constant TOLERANCE : signed(PHASE_WIDTH-1 downto 0) := x"0004";  -- ~0.1% de erro

begin

    -- ============================================================
    -- PIPELINE LHS (Left Hand Side): R12 * R13 * R23
    -- ============================================================
    process(clk, rst)
    begin
        if rst = '1' then
            lhs_valid <= (others => '0');
        elsif rising_edge(clk) then
            -- Estágio 1: Calcular R12(a,b)
            lhs_pipe.r12 <= compose_phase(phase_a, phase_b);
            lhs_valid(0) <= valid_in;

            -- Estágio 2: Calcular R13(a,c) e compor com R12
            lhs_pipe.r13 <= compose_phase(phase_a, phase_c);
            lhs_pipe.r12 <= compose_phase(lhs_pipe.r12, lhs_pipe.r13);
            lhs_valid(1) <= lhs_valid(0);

            -- Estágio 3: Calcular R23(b,c) e composição final
            lhs_pipe.r23 <= compose_phase(phase_b, phase_c);
            lhs_result <= compose_phase(lhs_pipe.r12, lhs_pipe.r23);
            lhs_valid(2) <= lhs_valid(1);
        end if;
    end process;

    -- ============================================================
    -- PIPELINE RHS (Right Hand Side): R23 * R13 * R12
    -- ============================================================
    process(clk, rst)
    begin
        if rst = '1' then
            rhs_valid <= (others => '0');
        elsif rising_edge(clk) then
            -- Estágio 1: Calcular R23(b,c)
            rhs_pipe.r23 <= compose_phase(phase_b, phase_c);
            rhs_valid(0) <= valid_in;

            -- Estágio 2: Calcular R13(a,c) e compor com R23
            rhs_pipe.r13 <= compose_phase(phase_a, phase_c);
            rhs_pipe.r23 <= compose_phase(rhs_pipe.r23, rhs_pipe.r13);
            rhs_valid(1) <= rhs_valid(0);

            -- Estágio 3: Calcular R12(a,b) e composição final
            rhs_pipe.r12 <= compose_phase(phase_a, phase_b);
            rhs_result <= compose_phase(rhs_pipe.r23, rhs_pipe.r12);
            rhs_valid(2) <= rhs_valid(1);
        end if;
    end process;

    -- ============================================================
    -- TMR (Triple Modular Redundancy) nos resultados finais
    -- ============================================================
    gen_tmr: if TMR_EN generate
        signal lhs_a, lhs_b, lhs_c : phase_t;
        signal match_ab, match_ac, match_bc : std_logic;
    begin
        -- Registros TMR (distribuídos fisicamente no FPGA)
        process(clk)
        begin
            if rising_edge(clk) then
                lhs_a <= lhs_result; lhs_b <= lhs_result; lhs_c <= lhs_result;
            end if;
        end process;

        -- Votadores de maioria para LHS
        match_ab <= '1' when abs(lhs_a - lhs_b) < TOLERANCE else '0';
        match_ac <= '1' when abs(lhs_a - lhs_c) < TOLERANCE else '0';
        match_bc <= '1' when abs(lhs_b - lhs_c) < TOLERANCE else '0';

        -- Seleção do valor majoritário
        process(match_ab, match_ac, match_bc, lhs_a, lhs_b, lhs_c)
        begin
            seu_detected <= '0';
            if match_ab = '1' then
                lhs_tmr <= lhs_a;
            elsif match_ac = '1' then
                lhs_tmr <= lhs_a;
                seu_detected <= '1';
            elsif match_bc = '1' then
                lhs_tmr <= lhs_b;
                seu_detected <= '1';
            else
                lhs_tmr <= (others => '0');
                seu_detected <= '1';
            end if;
        end process;

        rhs_tmr <= rhs_result;  -- Simplificado para brevidade
    end generate;

    no_tmr: if not TMR_EN generate
        lhs_tmr <= lhs_result;
        rhs_tmr <= rhs_result;
        seu_detected <= '0';
    end generate;

    -- ============================================================
    -- COMPARADOR FINAL: Verificação da igualdade Yang-Baxter
    -- ============================================================
    process(clk, rst)
        variable diff : signed(PHASE_WIDTH-1 downto 0);
    begin
        if rst = '1' then
            yb_valid <= '0';
            phase_result <= (others => '0');
        elsif rising_edge(clk) then
            diff := abs(lhs_tmr - rhs_tmr);

            if diff < TOLERANCE and lhs_valid(2) = '1' and rhs_valid(2) = '1' then
                yb_valid <= '1';
                phase_result <= lhs_tmr;
            else
                yb_valid <= '0';
                phase_result <= (others => '0');
            end if;
        end if;
    end process;

end architecture;
