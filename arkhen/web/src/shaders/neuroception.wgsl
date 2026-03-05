// arkhen_neuroception.wgsl
struct KatharosVector {
    bio: f32,
    aff: f32,
    soc: f32,
    cog: f32,
    q_permeability: f32,
    padding: f32, // Ensure 16-byte alignment
};

@group(0) @binding(0) var<storage, read_write> vk_state: array<KatharosVector>;
@group(0) @binding(1) var<storage, read> vk_ref: KatharosVector;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if (id >= arrayLength(&vk_state)) {
        return;
    }

    var local_vk = vk_state[id];

    // Cálculo do Desvio Homeostático ΔK
    let diff = vec4<f32>(
        local_vk.bio - vk_ref.bio,
        local_vk.aff - vk_ref.aff,
        local_vk.soc - vk_ref.soc,
        local_vk.cog - vk_ref.cog
    );

    // Pesos W: Bio(0.35), Aff(0.30), Soc(0.20), Cog(0.15)
    let weights = vec4<f32>(0.35, 0.30, 0.20, 0.15);
    let weighted_diff = diff * weights;
    let delta_k = length(weighted_diff);

    // Atualiza a Permeabilidade Q(t) no buffer WebGPU
    // Elena (H) is the relational sustainability constant, here simplified.
    local_vk.q_permeability = max(0.0, 1.0 - delta_k);

    // Simple drift towards reference or φ-based orbit could be added here

    vk_state[id] = local_vk;
}
