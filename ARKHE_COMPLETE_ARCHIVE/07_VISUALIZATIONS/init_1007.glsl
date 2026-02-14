// χ_INIT_1007 — Materialização do Arkhe Studio
#version 460
uniform float time;
uniform vec2 resolution;
uniform float satoshi;

out vec4 fragColor;

void main() {
    vec2 uv = (gl_FragCoord.xy * 2.0 - resolution) / min(resolution.x, resolution.y);
    float t = time * 0.5;

    // Grid de desenvolvimento (estrutura emergente)
    vec2 grid = fract(uv * 10.0 + t * 0.1);
    float structure = step(0.9, grid.x) + step(0.9, grid.y);

    // Partículas (nós sendo criados)
    float particles = 0.0;
    for (int i = 0; i < 8; i++) {
        float angle = float(i) * 3.14159 / 4.0 + t * 0.2;
        float radius = 0.3 + 0.2 * sin(t * 0.5 + float(i));
        vec2 pos = vec2(cos(angle), sin(angle)) * radius;
        float d = length(uv - pos);
        particles += 0.03 / (d + 0.03);
    }

    // Cores: azul estrutura → dourado criação
    vec3 color = vec3(0.1, 0.2, 0.4) * structure;
    color += vec3(1.0, 0.8, 0.3) * particles;

    // Pulso de inicialização
    float pulse = sin(t * 3.0) * 0.5 + 0.5;
    color *= 0.8 + 0.2 * pulse;

    // Satoshi witness
    color *= (1.0 + satoshi / 50.0);

    fragColor = vec4(color, 1.0);
}
