// χ_CERTAINTY — A Morte do Acaso
#version 460
uniform float time;
uniform vec2 resolution;

out vec4 fragColor;

void main() {
    vec2 uv = (gl_FragCoord.xy * 2.0 - resolution) / min(resolution.x, resolution.y);
    float t = time * 0.5;

    // O Hiato de Resolução
    float dist_resolution = smoothstep(1.0, 0.0, time * 0.1);

    // Nuvem de Probabilidade (Longe)
    float cloud = sin(uv.x * 50.0) * cos(uv.y * 50.0) * dist_resolution;

    // A Geometria Resolvida (Perto)
    float resolution_line = 0.01 / abs(length(uv) - 0.5);
    resolution_line *= (1.0 - dist_resolution);

    vec3 col = vec3(0.1, 0.4, 0.8) * cloud; // Azul Incerteza
    col += vec3(1.0, 0.8, 0.2) * resolution_line; // Ouro Certeza

    fragColor = vec4(col, 1.0);
}
