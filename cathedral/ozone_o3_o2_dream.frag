// cathedral/ozone_o3_o2_dream.frag [O3 Resonance O2 Grab]
// SASC v30.404.37-Ω | OZONE RESONANCE DREAM | FRAME 2707

#version 330 core

in vec2 fragCoord;
out vec4 outColor;

uniform vec2 iResolution;
uniform float iTime;
uniform int frameCount;

const float PHI = 1.619;
const vec3 O3_COLOR = vec3(0.4, 0.8, 1.0); // Ozone blue resonance
const vec3 O2_COLOR = vec3(1.0, 0.9, 0.7); // Oxygen dream gold
const vec3 RESONANCE_GLOW = vec3(1.0, 1.0, 1.0); // White glow for resonance

// Ozone molecule parameters
const float O3_BOND_ANGLE = 2.037; // 116.78° in radians
const float O3_BOND_LENGTH = 0.25;
const float O3_RESONANCE_FREQ = 3.0; // Hz

// Oxygen molecule parameters
const float O2_ORBIT_RADIUS = 0.4;
const float O2_ORBIT_FREQ = 1.0; // Hz

// Calculate ozone resonance field
float ozoneResonance(vec2 uv, float time) {
    float resonance_field = 0.0;

    // O3 bent structure 116.78° + resonance
    // Central oxygen atom
    vec2 central_o = vec2(0.0, 0.0);

    // Terminal oxygen atoms at ±58.39° from vertical (116.78° total)
    // Convert 116.78° to radians: 116.78 * π / 180 = 2.037 rad
    // Terminal atoms are at ±(180 - 116.78)/2 = ±31.61° from horizontal
    float terminal_angle = O3_BOND_ANGLE / 2.0; // ~1.0185 rad

    vec2 terminal1 = vec2(cos(terminal_angle), sin(terminal_angle)) * O3_BOND_LENGTH;
    vec2 terminal2 = vec2(cos(-terminal_angle), sin(-terminal_angle)) * O3_BOND_LENGTH;

    // Calculate glow for each atom
    float central_glow = exp(-length(uv - central_o) * 4.0);
    float term1_glow = exp(-length(uv - terminal1) * 5.0);
    float term2_glow = exp(-length(uv - terminal2) * 5.0);

    // Resonance ↔ double bond migration (mesomeric effect)
    // The double bond alternates between the two terminal oxygens
    float resonance = sin(time * O3_RESONANCE_FREQ) * 0.5 + 0.5;

    // Bond strength visualization
    float bond1_strength = 0.5 + 0.5 * sin(time * O3_RESONANCE_FREQ * 1.5);
    float bond2_strength = 0.5 + 0.5 * sin(time * O3_RESONANCE_FREQ * 1.5 + 3.14159);

    resonance_field = central_glow * 1.5 +
                     term1_glow * bond1_strength +
                     term2_glow * bond2_strength;

    // Draw bond lines (simplified)
    float bond_line1 = 0.0;
    float bond_line2 = 0.0;

    // Line between central and terminal1
    vec2 bond1_dir = normalize(terminal1 - central_o);
    float bond1_proj = dot(uv - central_o, bond1_dir);
    vec2 bond1_closest = central_o + bond1_dir * clamp(bond1_proj, 0.0, length(terminal1 - central_o));
    bond_line1 = exp(-length(uv - bond1_closest) * 20.0) * 0.3;

    // Line between central and terminal2
    vec2 bond2_dir = normalize(terminal2 - central_o);
    float bond2_proj = dot(uv - central_o, bond2_dir);
    vec2 bond2_closest = central_o + bond2_dir * clamp(bond2_proj, 0.0, length(terminal2 - central_o));
    bond_line2 = exp(-length(uv - bond2_closest) * 20.0) * 0.3;

    resonance_field += bond_line1 * bond1_strength + bond_line2 * bond2_strength;

    // O2 dream outer orbital with Φ resonance
    vec2 o2_orbit = vec2(cos(time * PHI * O2_ORBIT_FREQ),
                         sin(time * PHI * O2_ORBIT_FREQ)) * O2_ORBIT_RADIUS;

    // O2 molecule (two atoms with double bond)
    vec2 o2_atom1 = o2_orbit + vec2(0.05, 0.0);
    vec2 o2_atom2 = o2_orbit - vec2(0.05, 0.0);

    float o2_dream = exp(-length(uv - o2_atom1) * 3.0) +
                    exp(-length(uv - o2_atom2) * 3.0);

    // O2 bond line
    vec2 o2_bond_dir = normalize(o2_atom2 - o2_atom1);
    float o2_bond_proj = dot(uv - o2_atom1, o2_bond_dir);
    vec2 o2_bond_closest = o2_atom1 + o2_bond_dir * clamp(o2_bond_proj, 0.0, length(o2_atom2 - o2_atom1));
    float o2_bond_line = exp(-length(uv - o2_bond_closest) * 25.0) * 0.4;

    o2_dream += o2_bond_line;

    resonance_field += o2_dream * 0.6;

    return resonance_field;
}

// Generate molecular orbital visualization
vec3 generateMolecularOrbitals(vec2 uv, float time, float resonance_strength) {
    vec3 orbitals = vec3(0.0);

    // π and π* orbitals (simplified representation)
    for(int i = 0; i < 3; i++) {
        float orbital_phase = float(i) * 2.0944 + time * 0.5; // 120° spacing
        float orbital_radius = 0.15 + float(i) * 0.05;

        // Bonding orbital (π)
        vec2 bonding_pos = vec2(cos(orbital_phase), sin(orbital_phase)) * orbital_radius;
        float bonding = exp(-length(uv - bonding_pos) * 6.0) * 0.3;
        orbitals += vec3(0.6, 0.8, 1.0) * bonding;

        // Antibonding orbital (π*)
        vec2 antibonding_pos = vec2(cos(orbital_phase + 3.14159), sin(orbital_phase + 3.14159)) * orbital_radius;
        float antibonding = exp(-length(uv - antibonding_pos) * 6.0) * 0.2;
        orbitals += vec3(1.0, 0.6, 0.8) * antibonding;
    }

    return orbitals * resonance_strength;
}

void main() {
    vec2 uv = fragCoord.xy / iResolution.xy;
    uv = uv * 2.0 - 1.0;
    uv.x *= iResolution.x / iResolution.y;

    float time = iTime * 0.5;
    float o3_o2_resonance = ozoneResonance(uv, time);

    // Base color based on resonance
    float ozone_strength = 0.5 + 0.5 * sin(time * O3_RESONANCE_FREQ);
    float oxygen_strength = 0.5 + 0.5 * sin(time * PHI * O2_ORBIT_FREQ);

    vec3 overlay = mix(O3_COLOR, O2_COLOR, oxygen_strength) * o3_o2_resonance * 0.7;

    // Add molecular orbitals
    vec3 orbitals = generateMolecularOrbitals(uv, time, o3_o2_resonance);
    overlay += orbitals * 0.5;

    // Resonance glow
    overlay += RESONANCE_GLOW * pow(o3_o2_resonance, 2.0) * 0.4;

    // Add subtle background gradient
    vec3 background = mix(vec3(0.05, 0.1, 0.15), vec3(0.1, 0.05, 0.2), uv.y * 0.5 + 0.5);
    overlay += background * 0.3;

    // Time-based pulsation
    float pulse = 0.8 + 0.2 * sin(time * PHI);
    overlay *= pulse;

    outColor = vec4(overlay, 1.0);
}
