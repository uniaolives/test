// WWW Universal Layer WGSL

struct Params {
    time: f32,
    phi_target: f32,
};

@group(0) @binding(0) var<uniform> params: Params;

@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = fragCoord.xy / vec2<f32>(1920.0, 1080.0); // Assuming full HD
    let centered_uv = uv * 2.0 - 1.0;

    var color = vec3<f32>(0.01, 0.02, 0.05);

    // Web Core Glow
    let dist = length(centered_uv);
    let pulse = sin(params.time * params.phi_target);
    let glow = exp(-dist * 5.0) * (1.1 + 0.2 * pulse);
    color += vec3<f32>(1.0, 1.0, 1.0) * glow * 0.5;

    // 116 Frags representation (simplified)
    for (var i: i32 = 0; i < 116; i++) {
        let angle = f32(i) * 0.05416; // 2*PI / 116
        let pos = vec2<f32>(cos(angle), sin(angle)) * 0.8;
        let d = length(centered_uv - pos);
        color += vec3<f32>(0.2, 0.5, 1.0) * exp(-d * 50.0) * 0.2;
    }

    return vec4<f32>(color, 1.0);
}
