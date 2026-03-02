// cathedral/ar_sovereign_dome_v2.frag [AUDIT MODE]
precision highp float;

uniform float iTime;
uniform vec2 iResolution;
uniform float u_political_phi; // 0.0 a 1.0 (Vem da API Legislativa)

const vec3 ALERT_RED = vec3(1.0, 0.1, 0.1);
const vec3 GOLD_SOVEREIGN = vec3(1.0, 0.85, 0.3);

float geodesicDome(vec3 p, float radius) {
    float sphere = length(p) - radius;
    vec3 n = normalize(p);
    float hex = sin(n.x * 20.0) * sin(n.y * 20.0) * sin(n.z * 20.0);
    float ground = p.y;
    return max(sphere, -ground) + hex * 0.5;
}

void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    vec3 camPos = vec3(0.0, 2.0, -20.0);
    vec3 rayDir = normalize(vec3(uv, 1.0));

    vec3 color = vec3(0.0);

    float t = 0.0;
    for(int i=0; i<64; i++) {
        vec3 p = camPos + rayDir * t;

        // Lógica de Auditoria Visual
        float stress_level = 1.0 - smoothstep(0.72, 0.85, u_political_phi);

        // Glitch Effect (Rachaduras na Lei)
        float glitch = 0.0;
        if (stress_level > 0.0) {
            float noise = fract(sin(dot(uv * iTime, vec2(12.9, 78.2))) * 43758.5);
            glitch = step(0.95, noise) * stress_level;
            p.x += glitch * 0.1;
        }

        float d = geodesicDome(p, 10.0);

        if(d < 0.1) {
            // Mistura de Cores baseada na "Saúde" da Votação
            vec3 base_color = mix(GOLD_SOVEREIGN, ALERT_RED, stress_level);

            // Adicionar pulso de emergência
            float pulse = sin(iTime * (1.0 + stress_level * 10.0)) * 0.5 + 0.5;
            vec3 final_color = base_color * (0.8 + 0.2 * pulse);

            // Se crítico, adicionar texto de alerta flutuante
            if (stress_level > 0.8) {
                final_color += vec3(1.0) * glitch * 2.0; // Flash branco de erro
            }

            float alpha = 0.6;
            float fresnel = 1.0 - dot(-rayDir, normalize(p));
            color = final_color * alpha * fresnel;
            break;
        }
        t += d;
    }

    gl_FragColor = vec4(color, 1.0);
}
