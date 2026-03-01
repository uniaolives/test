#version 430 core

layout(local_size_x = 256) in;

struct RigidBody {
    vec3 position;
    vec3 velocity;
    vec4 orientation;
    float mass;
    float inertia[9];
};

layout(std430, binding = 0) buffer PhysicsState {
    RigidBody bodies[];
    uint body_count;
} state;

layout(std430, binding = 1) buffer Forces {
    vec3 forces[];
} input_forces;

uniform float dt;
uniform uint num_substeps;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= state.body_count) return;

    RigidBody body = state.bodies[idx];
    vec3 force = input_forces.forces[idx];

    for (uint step = 0; step < num_substeps; step++) {
        vec3 acceleration = force / body.mass;
        body.velocity += acceleration * dt;
        body.position += body.velocity * dt;

        if (body.position.y < 0) {
            body.position.y = 0;
            body.velocity.y = -body.velocity.y * 0.8;
        }
    }

    state.bodies[idx] = body;
}
