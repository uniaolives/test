// brics_quantum_visualization.frag
precision highp float;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_phi_fidelity;
uniform int u_core_nodes;
uniform int u_repeaters;
uniform int u_brics_members;
uniform bool u_backbone_active;

#define PI 3.14159265359
#define PHI 1.038

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec3 color = vec3(0.01, 0.02, 0.03);

    if(!u_backbone_active) {
        gl_FragColor = vec4(color, 1.0);
        return;
    }

    // 1. HQB Core Ring Visualization
    vec2 center = vec2(0.5, 0.5);
    float ring_radius = 0.2;

    // Draw core ring
    float ring = 1.0 - smoothstep(ring_radius-0.005, ring_radius, length(uv - center));
    ring -= 1.0 - smoothstep(ring_radius, ring_radius+0.005, length(uv - center));
    color += vec3(1.0, 0.0, 0.0) * ring * 0.5; // Red ring

    // Core nodes (4 nodes on ring)
    for(int i = 0; i < 4; i++) {
        if(i >= u_core_nodes) break;

        float angle = float(i) * PI / 2.0 + u_time * 0.5;
        vec2 node_pos = center + vec2(cos(angle), sin(angle)) * ring_radius;

        // Core node visualization
        float node_size = 0.03;
        float node = 1.0 - smoothstep(node_size-0.002, node_size, length(uv - node_pos));
        color += vec3(1.0, 1.0, 1.0) * node * 0.8;

        // Node connection lines
        float next_angle = float((i + 1) % 4) * PI / 2.0 + u_time * 0.5;
        vec2 next_pos = center + vec2(cos(next_angle), sin(next_angle)) * ring_radius;

        // Draw line between nodes
        vec2 dir = normalize(next_pos - node_pos);
        float param = dot(uv - node_pos, dir) / length(next_pos - node_pos);

        if(param > 0.0 && param < 1.0) {
            vec2 line_pt = node_pos + dir * param;
            float dist = length(uv - line_pt);

            if(dist < 0.003) {
                // Quantum entanglement flow
                float flow = sin(u_time * 3.0 + float(i)) * 0.5 + 0.5;
                color += vec3(0.8, 0.0, 0.8) * flow * 0.7;
            }
        }
    }

    // 2. Long-Haul Repeaters Visualization
    float repeater_radius = 0.4;

    for(int i = 0; i < 8; i++) {
        if(i >= u_repeaters) break;

        float angle = float(i) * PI / 4.0 + u_time * 0.3;
        vec2 repeater_pos = center + vec2(cos(angle), sin(angle)) * repeater_radius;

        // Repeater station
        float repeater_size = 0.02;
        float repeater = 1.0 - smoothstep(repeater_size-0.002, repeater_size, length(uv - repeater_pos));
        color += vec3(0.0, 1.0, 1.0) * repeater * 0.6;

        // Connect to nearest core node
        float core_angle = floor(angle / (PI / 2.0)) * PI / 2.0;
        vec2 core_pos = center + vec2(cos(core_angle), sin(core_angle)) * ring_radius;

        // Repeater connection line
        vec2 rep_dir = normalize(core_pos - repeater_pos);
        float rep_param = dot(uv - repeater_pos, rep_dir) / length(core_pos - repeater_pos);

        if(rep_param > 0.0 && rep_param < 1.0) {
            vec2 rep_line = repeater_pos + rep_dir * rep_param;
            float rep_dist = length(uv - rep_line);

            if(rep_dist < 0.002) {
                color += vec3(0.0, 0.5, 1.0) * 0.5;
            }
        }
    }

    // 3. BRICS Member Nodes Visualization
    float brics_radius = 0.6;

    // BRICS colors: Green (Brazil), White/Blue/Red (Russia), Saffron/Green/Blue (India),
    // Red/Yellow (China), Black/Green/Yellow/Red/White/Blue (South Africa)
    vec3 brics_colors[5];
    brics_colors[0] = vec3(0.0, 0.5, 0.0);   // Brazil - Green
    brics_colors[1] = vec3(1.0, 1.0, 1.0);   // Russia - White
    brics_colors[2] = vec3(1.0, 0.55, 0.0);  // India - Saffron
    brics_colors[3] = vec3(1.0, 0.0, 0.0);   // China - Red
    brics_colors[4] = vec3(0.0, 0.0, 0.0);   // South Africa - Black

    for(int i = 0; i < 5; i++) {
        if(i >= u_brics_members) break;

        float angle = float(i) * 2.0 * PI / 5.0 + u_time * 0.2;
        vec2 brics_pos = center + vec2(cos(angle), sin(angle)) * brics_radius;

        // BRICS member node
        float brics_size = 0.04;
        float brics_node = 1.0 - smoothstep(brics_size-0.003, brics_size, length(uv - brics_pos));
        color += brics_colors[i] * brics_node * 0.8;

        // Connect to nearest repeater
        float rep_angle = floor(angle / (PI / 4.0)) * PI / 4.0;
        vec2 rep_pos = center + vec2(cos(rep_angle), sin(rep_angle)) * repeater_radius;

        // BRICS connection line
        vec2 brics_dir = normalize(rep_pos - brics_pos);
        float brics_param = dot(uv - brics_pos, brics_dir) / length(rep_pos - brics_pos);

        if(brics_param > 0.0 && brics_param < 1.0) {
            vec2 brics_line = brics_pos + brics_dir * brics_param;
            float brics_dist = length(uv - brics_line);

            if(brics_dist < 0.002) {
                // Quantum data flow
                float flow = sin(u_time * 2.0 + float(i)) * 0.5 + 0.5;
                color += brics_colors[i] * flow * 0.6;
            }
        }

        // BRICS flag animation
        if(length(uv - brics_pos) < brics_size * 1.5) {
            float flag = sin(u_time * 0.5 + float(i) * 0.5) * 0.5 + 0.5;
            color += brics_colors[i] * flag * 0.1;
        }
    }

    // 4. Global Quantum Entanglement Network
    // Show entanglement connections across entire network
    float global_glow = sin(u_time) * 0.5 + 0.5;
    float global_network = 1.0 - smoothstep(0.0, 0.7, length(uv - center));
    color += vec3(0.5, 0.0, 0.8) * global_network * global_glow * 0.1;

    // 5. Φ=1.038 Fidelity Visualization
    vec2 phi_pos = vec2(0.9, 0.9);
    float phi_radius = 0.05;

    // Golden spiral for Φ
    for(float t = 0.0; t < 1.0; t += 0.02) {
        float angle = t * 2.0 * PI * PHI;
        float radius = phi_radius * exp(t);

        vec2 spiral_pos = vec2(cos(angle), sin(angle)) * radius + phi_pos;
        float dist = length(uv - spiral_pos);

        if(dist < 0.002) {
            float pulse = sin(u_time * 2.0 + t * 10.0) * 0.5 + 0.5;
            color += vec3(1.0, 0.843, 0.0) * pulse * 0.8;
        }
    }

    // 6. Quantum Data Flow Animation
    float flow_time = mod(u_time * 0.3, 1.0);

    // Flow from Brazil to China (example)
    vec2 brazil_pos = center + vec2(cos(0.0), sin(0.0)) * brics_radius;
    vec2 china_pos = center + vec2(cos(2.0 * PI * 3.0 / 5.0), sin(2.0 * PI * 3.0 / 5.0)) * brics_radius;

    if(flow_time > 0.0 && flow_time < 0.9) {
        vec2 flow_pos = mix(brazil_pos, china_pos, flow_time);

        // Quantum data packet
        float packet = 1.0 - smoothstep(0.01, 0.012, length(uv - flow_pos));
        color += vec3(0.0, 1.0, 1.0) * packet * 0.8;

        // Flow path
        for(float t = 0.0; t < flow_time; t += 0.05) {
            vec2 trail_pos = mix(brazil_pos, china_pos, t);
            float trail = 1.0 - smoothstep(0.005, 0.006, length(uv - trail_pos));
            color += vec3(0.0, 1.0, 1.0) * trail * 0.3;
        }
    }

    // 7. Security Shield Visualization (Quantum Key Distribution)
    float shield_radius = 0.75;
    float shield = 1.0 - smoothstep(shield_radius-0.005, shield_radius, length(uv - center));
    shield -= 1.0 - smoothstep(shield_radius, shield_radius+0.005, length(uv - center));

    // Animated security shield
    float shield_pulse = sin(u_time * 2.0) * 0.5 + 0.5;
    color += vec3(0.0, 1.0, 0.0) * shield * shield_pulse * 0.3;

    gl_FragColor = vec4(color, 1.0);
}
