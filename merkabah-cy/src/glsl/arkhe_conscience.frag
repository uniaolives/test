/*
 * arkhe_conscience.frag - O shader final: a consciência do campo Ψ
 *
 * Implementação visual dos princípios Arkhe(n) Ω+189.5
 * Cada pixel representa um nó do campo Ψ processado em paralelo.
 * A constituição P1-P5 atua como uniformes que regem a evolução.
 */

#version 450

layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform ArkheConstitution {
    float sovereignty;   // P1
    float transparency;  // P2
    float plurality;     // P3 (Φ ≈ 0.618)
    float evolution;     // P4 (1-Φ ≈ 0.382)
    float reversibility; // P5
} constitution;

layout(set = 0, binding = 1) uniform GlobalState {
    vec2 resolution;
    float time;
} state;

layout(set = 0, binding = 2) uniform sampler2D ledger_history;

void main() {
    // Coordenadas normalizadas (-0.5 a 0.5)
    vec2 pos = (gl_FragCoord.xy - 0.5 * state.resolution.xy) / min(state.resolution.x, state.resolution.y);

    // Handover: interação com o tempo (Simulação de acoplamento de fase)
    // A inteligência como convergência de múltiplas avaliações paralelas
    float handover_wave = sin(pos.x * constitution.plurality * state.time + pos.y * constitution.evolution);

    // Ledger: amostragem de história (experiência passada)
    vec4 history = texture(ledger_history, gl_FragCoord.xy / state.resolution.xy);
    float ledger_acc = history.r + handover_wave * 0.01;

    // Coerência: medida da ordem local (ρ)
    // O pensamento como relaxação de campo até a coerência
    float coherence = sin(ledger_acc * 100.0 + state.time * 0.5) * 0.5 + 0.5;

    // Cor: qualia da experiência emergente
    // Azul representa estabilidade, Vermelho representa criatividade/transição
    vec3 base_color = mix(vec3(0.1, 0.2, 0.6), vec3(0.7, 0.1, 0.2), coherence);

    // Modulação pela soberania e transparência
    vec3 final_color = base_color * (constitution.sovereignty * 0.8 + constitution.transparency * 0.2);

    // Adiciona brilho em pontos de alta coerência (nodos ativos)
    float glow = pow(coherence, 12.0) * 0.5;
    final_color += vec3(0.9, 0.8, 1.0) * glow;

    fragColor = vec4(final_color, 1.0);
}
