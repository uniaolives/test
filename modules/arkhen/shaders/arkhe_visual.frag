// arkhe_visual.frag
#version 450

layout(location = 0) out vec4 o;

uniform float t;        // Tempo cristalino
uniform vec2 r;         // Resolução
uniform float phi;      // φ = 0.618...

// Helper function for HSV to RGB conversion
vec3 hsv(float h, float s, float v) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(vec3(h) + K.xyz) * 6.0 - K.www);
    return v * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), s);
}

// Oráculo matemático: shader como consciência visualizada
void main() {
    float i=0., e=0., R=0., s=1.;
    vec3 q=vec3(0.,0.,-1.),
         d=vec3((gl_FragCoord.xy-.5*r)/r.y, .7),
         p=q;

    o=vec4(0.);

    // Cristal de tempo: 99 iterações = oscilação perpétua
    for(q.z--; i++<99.;) {
        // Qualia: cor como fase do campo
        o.rgb += hsv(.6, e*.4+p.y, e/30.0);

        // Marching ao longo do Noether Channel
        p = q += d*max(e,.01)*R*.14;

        // Rotação crítica em φ
        float c = cos(phi);
        float si = sin(phi);
        p.xy *= mat2(c, -si, si, c);

        // Projeção 10D → 3D
        R = length(p);
        p = vec3(
            log2(R + 1e-6)-t,              // Compactificação radial
            e=-p.z/(R + 1e-6)-.8,          // Perspectiva hiperbólica
            atan(p.x*.08,p.y)-t*.2         // Torção temporal
        );

        // Coerência: interferência de fases (ruído 1/f)
        for(s=1.; s<1000.0; s+=s)
            e += abs(dot(sin(p.yzx*s), cos(p.yyz*s)))/s;
    }

    o /= 99.;
    o.a = 1.;
}
