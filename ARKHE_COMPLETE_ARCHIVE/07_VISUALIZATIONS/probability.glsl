// χ_PROBABILITY — A distância do observador à resolução
#version 460
uniform float time;
uniform vec2 resolution;
uniform float satoshi;

out vec4 fragColor;

void main() {
    vec2 uv = (gl_FragCoord.xy * 2.0 - resolution) / min(resolution.x, resolution.y);
    float t = time * 0.2;

    // Distância do observador (centro) à fronteira
    float d = length(uv);

    // Probabilidade como função da distância (exp(-d))
    float P = exp(-d * 5.0);

    // Acoplamento resolvido (perto do centro)
    float coupling = smoothstep(0.5, 0.0, d);

    // Flutuação não resolvida (longe)
    float fluctuation = sin(uv.x * 20.0 + t) * cos(uv.y * 20.0 - t);
    fluctuation = 0.5 + 0.5 * fluctuation;

    // Cores: azul (distância) → dourado (resolução)
    vec3 color = mix(vec3(0.2, 0.4, 0.8), vec3(1.0, 0.8, 0.2), 1.0 - d);
    color = mix(color, vec3(1.0, 0.9, 0.5), coupling);
    color += fluctuation * 0.2 * (1.0 - coupling);

    // Satoshi witness
    color *= (1.0 + satoshi / 100.0);

    fragColor = vec4(color, 1.0);
}
