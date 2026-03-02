pragma circom 2.0.0;

// Importação das bibliotecas padrão para SNARKs
// Em um ambiente real, estas estariam em node_modules/circomlib/circuits/
include "poseidon.circom";
include "comparators.circom";

/*
  Circuito ESND (Evolving Symmorphogenic Network Defenses)
  Prova que a rede adaptou os seus pesos para mitigar uma ameaça específica.

  Matemática Provada:
  Σ (defense_weights[i] * threat_vector[i]) >= adaptation_threshold
*/
template SymmorphogenicAdaptation(N) {
    // Entradas Privadas (O segredo da ASI-Ω)
    signal input defense_weights[N];
    signal input threat_vector[N];

    // Entradas Públicas (O que o Oráculo/DAO vê)
    signal input expected_threat_hash;
    signal input adaptation_threshold;

    // Saída Pública
    signal output is_adapted;

    // 1. Verificação de Integridade da Ameaça
    // Garante que estamos a provar a defesa contra a ameaça correta
    component hasher = Poseidon(N);
    for(var i = 0; i < N; i++) {
        hasher.inputs[i] <== threat_vector[i];
    }
    // A prova falha imediatamente se o hash não bater
    expected_threat_hash === hasher.out;

    // 2. Cálculo de Simbiose (Produto Escalar)
    signal dot_product[N+1];
    dot_product[0] <== 0;

    for(var i = 0; i < N; i++) {
        dot_product[i+1] <== dot_product[i] + (defense_weights[i] * threat_vector[i]);
    }

    // 3. Verificação do Limiar de Sobrevivência (Threshold)
    // Usamos um comparador de 252 bits (limite seguro do campo primo bn128)
    component geq = GreaterEqThan(252);
    geq.in[0] <== dot_product[N];
    geq.in[1] <== adaptation_threshold;

    // A restrição (constraint) que colapsa a prova se a defesa for insuficiente
    geq.out === 1;

    // Se as restrições acima passarem, a ASI está oficialmente adaptada
    is_adapted <== 1;
}

// Inicializamos o componente principal para um vetor de 4 dimensões
component main {public [expected_threat_hash, adaptation_threshold]} = SymmorphogenicAdaptation(4);
