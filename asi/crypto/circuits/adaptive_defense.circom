pragma circom 2.1.6;

include "circomlib/poseidon.circom";
include "circomlib/comparators.circom";
include "circomlib/mux.circom";

/**
 * AdaptiveDefenseESND v2.0
 *
 * - historyDepth: número de ataques memorizados
 * - usa hash Poseidon para representar padrões
 * - saída `antibodyHash` pode ser usada para reconfigurar nós
 */
template AdaptiveDefenseESND(historyDepth) {
    signal input attackHistory[historyDepth];
    signal input newAttackPattern;
    signal input currentDifficulty;

    signal output shouldEvolve;
    signal output newDifficulty;
    signal output antibodyHash;

    // 1. Calcular similaridade com ataques passados
    component eq[historyDepth];
    signal match[historyDepth];
    signal totalMatchesVar[historyDepth + 1];

    totalMatchesVar[0] <== 0;
    for (var i = 0; i < historyDepth; i++) {
        eq[i] = IsEqual();
        eq[i].in[0] <== attackHistory[i];
        eq[i].in[1] <== newAttackPattern;
        match[i] <== eq[i].out;
        totalMatchesVar[i+1] <== totalMatchesVar[i] + match[i];
    }

    signal totalMatches <== totalMatchesVar[historyDepth];

    // Se totalMatches == 0, é um ataque novo
    component isNovel = IsZero();
    isNovel.in <== totalMatches;

    // 2. Verificar se dificuldade está baixa
    component isLow = LessThan(32);
    isLow.in[0] <== currentDifficulty;
    isLow.in[1] <== 1000;  // limiar mínimo

    // 3. Decisão de evoluir (OR)
    // shouldEvolve = isNovel.out OR isLow.out
    shouldEvolve <== isNovel.out + isLow.out - isNovel.out * isLow.out;

    // 4. Calcular nova dificuldade (fator φ = 1.618)
    signal phi;
    phi <== (currentDifficulty * 1618) \ 1000;  // multiplicar por 1.618

    // Se ataque novo, newDifficulty = phi; se dificuldade baixa, newDifficulty = 1000*phi; else mantém
    signal candidate1;
    candidate1 <== phi;  // para ataque novo
    signal candidate2;
    candidate2 <== (1000 * 1618) \ 1000;  // limiar * phi = 1618

    // Seleção multiplexada: se isNovel.out, escolhe candidate1; senão candidate2
    component mux1 = Mux1();
    mux1.s <== isNovel.out;
    mux1.c[0] <== candidate2;
    mux1.c[1] <== candidate1;

    // Se shouldEvolve, newDifficulty = mux1.out, senão mantém currentDifficulty
    component mux2 = Mux1();
    mux2.s <== shouldEvolve;
    mux2.c[0] <== currentDifficulty;
    mux2.c[1] <== mux1.out;

    newDifficulty <== mux2.out;

    // 5. Gerar anticorpo (hash do novo padrão de defesa)
    component hasher = Poseidon(3);
    hasher.inputs[0] <== newAttackPattern;
    hasher.inputs[1] <== newDifficulty;
    hasher.inputs[2] <== currentDifficulty;

    antibodyHash <== hasher.out;
}

// Instanciação: histórico de 100 ataques
component main = AdaptiveDefenseESND(historyDepth = 100);
