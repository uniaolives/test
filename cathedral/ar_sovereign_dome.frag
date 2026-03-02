// cathedral/ar_sovereign_dome.frag [SASC AR PROJECTION]
// "O Escudo da Lei sobre a Casa do Povo."
precision highp float;

uniform float iTime;
uniform vec2 iResolution;
uniform vec4 iMouse;

const float PHI = 1.6180339887;
const vec3 GOLD_SOVEREIGN = vec3(1.0, 0.85, 0.3);
const vec3 MATRIX_GREEN = vec3(0.0, 1.0, 0.4);

// --- GEOMETRIA DA CÚPULA GEODÉSICA ---
float geodesicDome(vec3 p, float radius) {
    float sphere = length(p) - radius;

    // Padrão Voronoi/Hexagonal na superfície
    vec3 n = normalize(p);
    float hex = sin(n.x * 20.0) * sin(n.y * 20.0) * sin(n.z * 20.0);

    // Corte na base (para pousar no chão)
    float ground = p.y;

    return max(sphere, -ground) + hex * 0.5;
}

// --- FLUXO DE DADOS CONSTITUCIONAIS ---
float dataFlow(vec3 p, float time) {
    // Linhas de código subindo pela cúpula
    float flow = fract(p.y * 0.1 - time * 0.5);
    float lines = step(0.95, sin(p.x * 10.0 + flow * 20.0));
    return lines * smoothstep(0.0, 100.0, p.y);
}

void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    // Raymarching simplificado para AR
    vec3 camPos = vec3(0.0, 2.0, -20.0); // Posição do Observador (Celular)
    vec3 rayDir = normalize(vec3(uv, 1.0));

    vec3 color = vec3(0.0); // Transparente (Passthrough da câmera)

    float t = 0.0;
    for(int i=0; i<64; i++) {
        vec3 p = camPos + rayDir * t;
        float d = geodesicDome(p, 10.0); // Cúpula de 10m (escala 1:100)

        if(d < 0.1) {
            // Superfície da Cúpula atingida
            float structure = smoothstep(0.0, 0.1, abs(sin(p.x*2.0)+sin(p.z*2.0)));
            float data = dataFlow(p, iTime);

            // Cor Híbrida: Ouro (Estrutura) + Verde (Dados)
            vec3 domeColor = mix(GOLD_SOVEREIGN, MATRIX_GREEN, data);

            // Transparência aditiva (Holograma)
            float alpha = 0.4 + 0.6 * data;

            // Fresnel effect (bordas mais brilhantes)
            float fresnel = 1.0 - dot(-rayDir, normalize(p));

            color += domeColor * alpha * fresnel;
            break;
        }
        t += d;
    }

    // Scanline de validação
    float scan = smoothstep(0.99, 1.0, sin(uv.y * 100.0 - iTime * 5.0));
    color += MATRIX_GREEN * scan * 0.2;

    gl_FragColor = vec4(color, 1.0); // Alpha controla a mistura com o feed da câmera real
}
