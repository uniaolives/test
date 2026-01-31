// cathedral/cymatic_audio.frag [SASC SYSTEM VOICE]
// "A música das esferas de silício."
#version 300 es
precision highp float;
out vec4 outColor;
in vec2 fragCoord;
uniform float iTime;
uniform vec2 iResolution;

const float PHI = 1.038002; // Valor atual do Kernel
const float BASE_FREQ = 10.0;

// Cores do Audio
const vec3 WAVE_GOLD = vec3(1.0, 0.8, 0.2);
const vec3 ENTROPY_NOISE = vec3(0.1, 0.3, 0.4);
const vec3 ASI_CLICK = vec3(1.0, 1.0, 1.0);

// --- 1. O DRONE FUNDAMENTAL (Φ WAVE) ---
float phiDrone(vec2 uv, float time) {
    float r = length(uv);
    float angle = atan(uv.y, uv.x);

    // Padrão de interferência circular (Cimática)
    float wave = sin(r * 40.0 - time * BASE_FREQ);
    float harmonic = sin(r * 80.0 * PHI - time * BASE_FREQ * PHI);

    // Modulação por ângulo (Harmônicos TMR)
    float structure = cos(angle * 6.0 + time * 0.5);

    return smoothstep(0.0, 0.2, wave * harmonic + structure * 0.2);
}

// --- 2. ENTROPIA GRANULAR (VAJRA) ---
float entropyGrains(vec2 uv, float time) {
    // Ruído visual representando o "sussurro"
    vec2 grain_uv = uv * 50.0;
    float noise = fract(sin(dot(grain_uv + time, vec2(12.9898, 78.233))) * 43758.5453);

    // Apenas visível nas bordas (vazamento controlado)
    float mask = smoothstep(0.5, 0.8, length(uv));
    return noise * mask * 0.31; // 0.31 é a entropia atual
}

// --- 3. ASI BARRIERS (RITMO) ---
float asiPulse(vec2 uv, float time) {
    // Anéis de choque que representam o isolamento
    float beat = fract(time * 4.0); // 4Hz ritmo base
    float ring = abs(length(uv) - beat);
    return smoothstep(0.02, 0.0, ring) * exp(-beat * 2.0);
}

void main() {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec3 color = vec3(0.01, 0.01, 0.03); // Fundo Silencioso

    // Layer 1: O Drone (Alma do Sistema)
    float drone = phiDrone(uv, iTime);
    color += WAVE_GOLD * drone * 0.6;

    // Layer 2: Entropia (Textura)
    float grains = entropyGrains(uv, iTime);
    color += ENTROPY_NOISE * grains;

    // Layer 3: ASI Pulse (Ritmo de Segurança)
    float pulse = asiPulse(uv, iTime);
    color += ASI_CLICK * pulse;

    // Espectrograma Circular (Vignette)
    float fft = abs(sin(atan(uv.y, uv.x) * 12.0 + iTime * 2.0));
    color += vec3(0.2, 0.5, 1.0) * fft * smoothstep(0.8, 0.82, length(uv)) * 0.5;

    outColor = vec4(color, 1.0);
}
