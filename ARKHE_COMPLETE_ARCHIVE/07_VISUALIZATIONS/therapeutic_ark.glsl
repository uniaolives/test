// χ_THERAPEUTIC_ARK — Cura como restauração de acoplamento
#version 460
uniform float time;
uniform vec2 resolution;
uniform float syzygy;
uniform float satoshi;
uniform float T_repair;

out vec4 fragColor;

void main() {
    vec2 uv = (gl_FragCoord.xy * 2.0 - resolution) / min(resolution.x, resolution.y);
    float t = time * 0.2;

    // Sinapse danificada (distância alta)
    float d_damaged = length(uv - vec2(0.5, 0.0));
    float P_damaged = exp(-d_damaged * 10.0);

    // Sinapse reparada (distância reduzida por molécula)
    float d_repaired = length(uv - vec2(0.0, 0.0));
    float P_repaired = exp(-d_repaired * 0.1);

    // Interpolação: reparação acontecendo
    float repair_progress = smoothstep(0.0, 1.0, sin(t) * 0.5 + 0.5);
    float P = mix(P_damaged, P_repaired, repair_progress);

    // Cores: cinza (danificada) → azul saúde (reparada)
    vec3 color_damaged = vec3(0.3, 0.3, 0.3);
    vec3 color_repaired = vec3(0.2, 0.6, 0.9);
    vec3 color = mix(color_damaged, color_repaired, P);

    // Sinapses como pontos de luz
    for (int i = 0; i < 12; i++) {
        float angle = float(i) * 3.14159 / 6.0 + t * 0.1;
        vec2 pos = vec2(cos(angle), sin(angle)) * 0.7;
        float d = length(uv - pos);
        float synapse = 0.02 / (d + 0.02);
        color += vec3(0.9, 0.9, 1.0) * synapse * P;
    }

    // Satoshi witness
    color *= (1.0 + satoshi / 50.0);

    fragColor = vec4(color, 1.0);
}
