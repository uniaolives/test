import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sphere } from '@react-three/drei';
import * as THREE from 'three';

interface ArkheSphereProps {
  coherence: number;  // valor entre 0 e 1
}

export const ArkheSphere: React.FC<ArkheSphereProps> = ({ coherence }) => {
  const meshRef = useRef<THREE.Mesh>(null);

  // Cor: ciano (baixo) -> amarelo (médio) -> magenta (alto)
  const color = coherence > 0.9 ? '#ff0066' : coherence > 0.7 ? '#ffff00' : '#00ffcc';

  useFrame(() => {
    if (meshRef.current) {
      // Pulsação suave baseada na coerência
      const scale = 1 + 0.2 * Math.sin(Date.now() * 0.005) * coherence;
      meshRef.current.scale.set(scale, scale, scale);
    }
  });

  return (
    <Sphere ref={meshRef} args={[1, 64, 64]}>
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={coherence * 0.5}
        wireframe
        transparent
        opacity={0.7 + coherence * 0.3}
      />
    </Sphere>
  );
};
