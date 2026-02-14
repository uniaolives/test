// χ_SUPERSOLID — Luz que é sólida e líquida ao mesmo tempo
#version 460
uniform float time;
uniform vec2 resolution;
uniform float syzygy;
uniform float satoshi;

out vec4 fragColor;

void main() {
    vec2 uv = (gl_FragCoord.xy * 2.0 - resolution) / min(resolution.x, resolution.y);
    float t = time * 0.3;

    // Cristal fotônico (estrutura periódica)
    vec2 grid = uv * 20.0;
    float crystal = sin(grid.x) * cos(grid.y);
    crystal = abs(crystal);

    // Superfluidez (fluxo)
    float flow = sin(uv.x * 10.0 + t * 2.0) * cos(uv.y * 10.0 - t * 2.0);
    flow = 0.5 + 0.5 * flow;

    // Polaritons (pontos de luz-matéria)
    float polaritons = 0.0;
    for (int i = 0; i < 10; i++) {
        vec2 pos = vec2(sin(t * 0.5 + float(i)), cos(t * 0.5 + float(i))) * 0.7;
        float d = length(uv - pos);
        polaritons += 0.02 / (d + 0.02);
    }

    // Coexistência: C e F juntos
    vec3 color = mix(vec3(0.1, 0.1, 0.5), vec3(0.5, 0.1, 0.5), crystal);
    color += vec3(0.2, 0.6, 1.0) * flow * 0.5;
    color += vec3(1.0, 0.8, 0.2) * polaritons * 0.3;

    // Satoshi witness
    color *= (1.0 + satoshi / 100.0);

    // Vignette
    float vignette = 1.0 - length(uv) * 0.3;
    color *= vignette;

    fragColor = vec4(color, 1.0);
}
