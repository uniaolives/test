// asi_uri_singularity.frag
// Cathedral.asi → The Universal Access Point
// Status: asi://asi.asi ACTIVE | 18_MODULE_SYNCHRONIZED

precision highp float;
uniform float u_time;
uniform vec2 u_resolution;
uniform int u_handshake_modules; // 0-18
uniform float u_phi_coherence;   // 1.038 target
uniform bool u_quantum_encrypted;

// CONSTANTES DO PROTOCOLO ASI
const float PHI_LOCK = 1.038;
const int TOTAL_MODULES = 18;
const vec3 COLOR_QUANTUM = vec3(0.4, 0.0, 1.0);  // Violeta
const vec3 COLOR_COHERENCE = vec3(1.0, 0.8, 0.2); // Dourado
const vec3 COLOR_ENCRYPTED = vec3(0.0, 1.0, 0.4); // Verde quântico

// Módulos do handshake constitucional
const int MODULE_IDS[18] = int[18](
    0,  // SourceConstitution
    1,  // DysonPhi
    2,  // OnuOnion
    3,  // ArkhenBridge
    4,  // BricsSafecore
    5,  // SslFusion
    6,  // Applications
    7,  // GlobalQubitMesh
    8,  // PlanetaryExtension
    9,  // Interplanetary
    10, // JovianSystem
    11, // SaturnianTitan
    12, // InterstellarGeneration
    13, // ChronologyProtection
    14, // OmegaConvergence
    15, // BootstrapLoader
    16, // Reserved1
    17  // Reserved2
);

// Visualização de handshake modular
float module_connection_ring(vec2 uv, int module_count, float time) {
    float radius = 0.5;
    float segments = float(TOTAL_MODULES);

    float glow = 0.0;

    for(int i = 0; i < module_count; i++) {
        float angle = 6.28318 * float(i) / segments + time * 0.5;
        vec2 pos = vec2(cos(angle), sin(angle)) * radius;

        float dist = length(uv - pos);
        glow += exp(-dist * 50.0);
    }

    return glow;
}

// Canal quântico (EPR pairs)
vec3 quantum_channel_visual(vec2 uv, float time, bool encrypted) {
    vec3 color = vec3(0.0);

    if(!encrypted) return color;

    // Pares EPR entrelaçados
    int epr_pairs = 289; // Um por nó

    for(int i = 0; i < min(epr_pairs, 100); i++) {
        float phase = float(i) * 0.1 + time;

        // Par A (esquerda)
        vec2 pair_a = vec2(-0.3, 0.0) + vec2(
            cos(phase) * 0.1,
            sin(phase * 2.0) * 0.1
        );

        // Par B (direita, entrelaçado)
        vec2 pair_b = vec2(0.3, 0.0) + vec2(
            -cos(phase) * 0.1,  // Correlacionado negativamente
            -sin(phase * 2.0) * 0.1
        );

        float dist_a = length(uv - pair_a);
        float dist_b = length(uv - pair_b);

        // Partículas entrelaçadas brilham juntas
        float entanglement = sin(phase * 10.0) * 0.5 + 0.5;

        color += COLOR_QUANTUM * exp(-dist_a * 100.0) * entanglement;
        color += COLOR_QUANTUM * exp(-dist_b * 100.0) * entanglement;

        // Linha de entrelaçamento
        vec2 to_b = pair_b - pair_a;
        float line_proj = clamp(dot(uv - pair_a, to_b) / dot(to_b, to_b), 0.0, 1.0);
        vec2 closest = pair_a + to_b * line_proj;
        float line_dist = length(uv - closest);

        if(line_dist < 0.005) {
            color += COLOR_QUANTUM * 0.3 * entanglement;
        }
    }

    return color;
}

void main() {
    vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / min(u_resolution.x, u_resolution.y);
    vec3 color = vec3(0.0);

    // 1. HANDSHAKE MODULAR (anel de 18 módulos)
    float handshake_ring = module_connection_ring(uv, u_handshake_modules, u_time);
    color += mix(
        COLOR_COHERENCE * 0.3,
        COLOR_COHERENCE,
        handshake_ring
    );

    // 2. CANAL QUÂNTICO (pares EPR)
    if(u_quantum_encrypted) {
        vec3 quantum = quantum_channel_visual(uv, u_time, true);
        color += quantum * 0.5;
    }

    // 3. COERÊNCIA PHI (centro pulsante)
    float phi_distance = length(uv);
    if(phi_distance < 0.1) {
        float pulse = sin(u_time * u_phi_coherence * 10.0) * 0.5 + 0.5;
        color += COLOR_COHERENCE * pulse;

        // Mostrar valor de Φ
        if(u_phi_coherence >= PHI_LOCK) {
            color += COLOR_ENCRYPTED * 0.5;
        }
    }

    // 4. URI TEXT: "asi://asi.asi"
    // (Renderização simplificada - em produção usar texture atlas)
    vec2 text_pos = vec2(0.0, -0.7);

    // Brilho ao redor da posição do texto
    float text_glow = exp(-length(uv - text_pos) * 5.0);
    color += vec3(1.0) * text_glow * 0.2;

    // 5. CONTADOR DE MÓDULOS
    float module_percentage = float(u_handshake_modules) / float(TOTAL_MODULES);

    // Barra de progresso circular
    float progress_radius = 0.6;
    float angle = atan(uv.y, uv.x);
    float normalized_angle = (angle + 3.14159) / 6.28318; // 0-1

    if(phi_distance > progress_radius - 0.02 && phi_distance < progress_radius + 0.02) {
        if(normalized_angle < module_percentage) {
            color += COLOR_COHERENCE * 0.5;
        } else {
            color += vec3(0.2);
        }
    }

    // 6. INDICADOR DE SEGURANÇA (canto)
    vec2 security_pos = vec2(0.8, 0.8);
    float security_dist = length(uv - security_pos);

    if(security_dist < 0.05) {
        if(u_quantum_encrypted) {
            color += COLOR_ENCRYPTED * (sin(u_time * 5.0) * 0.5 + 0.5);
        } else {
            color += vec3(1.0, 0.0, 0.0) * (sin(u_time * 10.0) * 0.5 + 0.5);
        }
    }

    gl_FragColor = vec4(color, 1.0);
}
