-- arkhe_crypto_ntt_butterfly.vhd
-- Operador Borboleta de Radix-2 para Transformada Teórica dos Números (Módulo Q=3329)

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity NTT_Butterfly is
    Port (
        clk     : in  STD_LOGIC;
        A_in    : in  SIGNED(15 downto 0); -- Coeficiente polinomial A
        B_in    : in  SIGNED(15 downto 0); -- Coeficiente polinomial B
        W_in    : in  SIGNED(15 downto 0); -- Twiddle factor (Fator de rotação)
        X_out   : out SIGNED(15 downto 0); -- Saída A + W*B mod Q
        Y_out   : out SIGNED(15 downto 0)  -- Saída A - W*B mod Q
    );
end NTT_Butterfly;

architecture Pipelined of NTT_Butterfly is
    constant Q_MODULUS : SIGNED(15 downto 0) := to_signed(3329, 16);

    -- Registos de Pipeline para atingir 200 MHz sem violar timing
    signal mult_res    : SIGNED(31 downto 0);
    signal wb_mod_q    : SIGNED(15 downto 0);
    signal a_reg       : SIGNED(15 downto 0);
begin
    process(clk)
    begin
        if rising_edge(clk) then
            -- Estágio 1: Multiplicação B * W
            mult_res <= B_in * W_in;
            a_reg <= A_in; -- Atrasa A para sincronizar

            -- Estágio 2: Redução Modular (W * B) mod Q
            -- (Num chip real, usa-se Redução de Barrett ou Montgomery para evitar divisões lógicas)
            wb_mod_q <= resize(mult_res mod Q_MODULUS, 16);

            -- Estágio 3: Adição e Subtração Modulares (A Borboleta)
            X_out <= (a_reg + wb_mod_q) mod Q_MODULUS;
            Y_out <= (a_reg - wb_mod_q + Q_MODULUS) mod Q_MODULUS; -- Evita números negativos
        end if;
    end process;
end Pipelined;
