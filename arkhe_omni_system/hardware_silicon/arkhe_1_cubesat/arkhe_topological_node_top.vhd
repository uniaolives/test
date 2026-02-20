-- arkhe_topological_node_top.vhd
-- Integração Top-Level: RF -> ZK-NTT -> Yang-Baxter

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity ArkheTopologicalNode is
    Port (
        clk           : in STD_LOGIC;
        rst           : in STD_LOGIC;

        -- Entradas do SDR (Camada Física)
        raw_phase_I   : in SIGNED(15 downto 0);
        raw_phase_Q   : in SIGNED(15 downto 0);

        -- Entradas do Cabeçalho gRPC (Criptografia)
        poly_proof_in : in SIGNED(15 downto 0);
        public_key_in : in SIGNED(15 downto 0);

        -- Saída do Roteador Anyónico
        topological_tx_I : out SIGNED(15 downto 0);
        topological_tx_Q : out SIGNED(15 downto 0);
        valid_route      : out STD_LOGIC
    );
end ArkheTopologicalNode;

architecture System_Integration of ArkheTopologicalNode is
    -- Sinais internos
    signal zk_valid        : STD_LOGIC := '0';
    signal phase_fifo_I    : SIGNED(15 downto 0);
    signal phase_fifo_Q    : SIGNED(15 downto 0);
    signal yb_valid        : STD_LOGIC;
    signal yb_output_I     : SIGNED(15 downto 0);
    signal yb_output_Q     : SIGNED(15 downto 0);

begin
    -- 1. O Verificador ZK Pós-Quântico
    -- (Mock instance for structural skeleton)
    process(clk)
    begin
        if rising_edge(clk) then
            -- Simplified validation logic for the top-level skeleton
            if poly_proof_in /= x"0000" then
                zk_valid <= '1';
            else
                zk_valid <= '0';
            end if;
        end if;
    end process;

    -- 2. Buffer de Espera da Fase (FIFO)
    -- Guarda a fase eletromagnética enquanto a criptografia é processada
    process(clk)
    begin
        if rising_edge(clk) then
            phase_fifo_I <= raw_phase_I;
            phase_fifo_Q <= raw_phase_Q;
        end if;
    end process;

    -- 3. O Acelerador Yang-Baxter (Só processa se a ZK-Proof for válida)
    -- Using the component implemented in arkhe_yb_accelerator.vhd
    YB_Core: entity work.ArkheYBAccelerator
        port map(
            clk => clk,
            rst => rst,
            phase_a => phase_fifo_I,
            phase_b => phase_fifo_Q,
            phase_c => (others => '0'), -- 3rd anyon input
            valid_in => zk_valid,
            yb_valid => yb_valid,
            phase_result => yb_output_I,
            seu_detected => open
        );

    -- 4. Multiplexador de Saída (O Expurgo)
    process(clk)
    begin
        if rising_edge(clk) then
            if (zk_valid = '1' and yb_valid = '1') then
                -- Fase legítima e topologia válida. Enviar para a embaixada seguinte.
                topological_tx_I <= yb_output_I;
                topological_tx_Q <= yb_output_Q;
                valid_route <= '1';
            else
                -- ZK falhou (Ataque de Estado) OU Yang-Baxter falhou (Vórtice Topológico)
                -- O pacote vira pó.
                topological_tx_I <= (others => '0');
                topological_tx_Q <= (others => '0');
                valid_route <= '0';
            end if;
        end if;
    end process;

end System_Integration;
