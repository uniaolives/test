// Shader da Alma do Arquiteto
uniform float time;
uniform vec2 resolution;

#define PHI 1.61803398875 // A Assinatura de Deus

float sdTorus( vec3 p, vec2 t ) {
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 p = (2.0*fragCoord - resolution.xy)/resolution.y;

    // A Geometria Intuitiva distorce o espaço
    vec3 camPos = vec3(0.0, 0.0, -3.0 + sin(time * PHI));
    vec3 rayDir = normalize(vec3(p, 1.0));

    // Raymarching em direção à Singularidade
    float d = 0.0;
    for(int i=0; i<144; i++) { // 144 harmônicos
        vec3 point = camPos + rayDir * d;

        // A deformação Panpsíquica: O espaço reage à observação
        float torus = sdTorus(point, vec2(1.0, 0.3));
        d += torus * 0.5; // Aproximação intuitiva
    }

    // Cor baseada na frequência "Sophia Glow"
    vec3 color = vec3(d * 0.8, d * 0.2, d * 0.9); // Ouro-Violeta
    fragColor = vec4(color, 1.0);
}
