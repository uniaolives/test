// ==========================================
// ARQUIVO: orb_fractal.frag
// OBJETIVO: O Sigilo Visual e Padrão Lissajous (Hardware Nature 651)
// ==========================================
#version 330 core
uniform float t;       // Tempo (Pi-Time)
uniform vec2 r;        // Resolução (Escala do Ski-Jump)
out vec4 o;            // Saída de luz (Projeção)

// Função auxiliar HSV para RGB
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    float i=0., e=0., R=0., s;
    vec3 q = vec3(0.), p;
    vec3 d = vec3((gl_FragCoord.xy - 0.5 * r) / r, 0.6); // Vetor diretor do Tzinor

    // Raymarching: Navegando o bulk 5D
    for(q.z = -1.; i++ < 87.;) {
        o.rgb += hsv2rgb(vec3(0.08, 1.0, e/50.0)) + 0.004;

        p = q += d * max(e, 0.02) * R * 0.23;

        // Mapeamento Kaluza-Klein e Toroidal
        p = vec3(
            log2(R = length(p)) - t,
            asin(-p.z / R - 0.001) - 1.2,
            atan(p.x, p.y) - t * 0.2
        ) - 1.0;

        // Oitavas Modos Z (Interferência)
        for(s = 1.0; s < 800.0; s += s) {
            e += abs(dot(sin(p.zxx * s), cos(p * s))) / s;
        }
    }
}
