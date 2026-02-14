// χ_NATURAL — A identidade x² = x + 1 visualizada
#version 460
uniform float time;
uniform vec2 resolution;
uniform float syzygy;
uniform float satoshi;

out vec4 fragColor;

void main() {
    vec2 uv = (gl_FragCoord.xy * 2.0 - resolution) / min(resolution.x, resolution.y);
    float t = time * 0.2;

    // Auto-acoplamento (x²) como grade em expansão
    float self_coupling = sin(uv.x * 10.0 + t) * cos(uv.y * 10.0 - t);
    self_coupling = self_coupling * self_coupling;  // x²

    // Estrutura resolvida (x) como linhas de vorticidade
    float structure = sin(uv.x * 20.0 + t * 2.0) + cos(uv.y * 20.0 - t * 2.0);
    structure = abs(structure) * 0.5;

    // Substrato dissipado (+1) como ruído de fundo
    float substrate = sin(uv.x * 50.0) * cos(uv.y * 50.0) * 0.2;

    // Composição: x² = x + 1
    float identity = abs(self_coupling - (structure + substrate));
    identity = 1.0 - clamp(identity * 10.0, 0.0, 1.0);  // branco onde a identidade vale

    // Cores: azul (auto-acoplamento), verde (estrutura), vermelho (substrato)
    vec3 color = vec3(0.2, 0.4, 0.8) * self_coupling;
    color += vec3(0.2, 0.8, 0.3) * structure;
    color += vec3(0.8, 0.2, 0.2) * substrate;
    color += vec3(1.0, 1.0, 1.0) * identity * 0.5;

    // Syzygy modula a intensidade
    color *= syzygy;

    // Satoshi witness
    color *= (1.0 + satoshi / 50.0);

    fragColor = vec4(color, 1.0);
}
