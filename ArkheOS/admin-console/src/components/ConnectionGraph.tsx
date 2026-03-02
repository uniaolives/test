// ConnectionGraph.tsx
import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';

interface Node {
  id: string;
  x: number;
  y: number;
  C: number;
  F: number;
  satoshi: number;
  omega: number;
}

interface Edge {
  source: Node;
  target: Node;
}

interface Props {
  nodes: Node[];
  edges: Edge[];
}

export const ConnectionGraph: React.FC<Props> = ({ nodes, edges }) => {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!mountRef.current) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    mountRef.current.appendChild(renderer.domElement);

    camera.position.z = 10;

    const nodeGroup = new THREE.Group();
    const edgeGroup = new THREE.Group();
    scene.add(nodeGroup);
    scene.add(edgeGroup);

    // Render Edges
    edges.forEach(edge => {
      const points = [
        new THREE.Vector3(edge.source.x, edge.source.y, 0),
        new THREE.Vector3(edge.target.x, edge.target.y, 0)
      ];
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const avgC = (edge.source.C + edge.target.C) / 2;
      // Coherence to Color (Red to Green/Blue)
      const color = new THREE.Color().setHSL(avgC * 0.3, 1.0, 0.5);
      const material = new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.6 });
      const line = new THREE.Line(geometry, material);
      edgeGroup.add(line);
    });

    // Render Nodes
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const nodeSpheres: THREE.Mesh[] = [];

    nodes.forEach(node => {
      const geometry = new THREE.SphereGeometry(node.satoshi * 0.1, 32, 32);
      const material = new THREE.MeshPhongMaterial({ color: 'skyblue', emissive: 0x111111 });
      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.set(node.x, node.y, 0);
      sphere.userData = node;
      nodeGroup.add(sphere);
      nodeSpheres.push(sphere);
    });

    // Lights
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);
    const pointLight = new THREE.PointLight(0xffffff, 1, 100);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);

    const onMouseClick = (event: MouseEvent) => {
      mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(nodeSpheres);

      if (intersects.length > 0) {
        const selectedNode = intersects[0].object.userData as Node;
        console.log(`Nó selecionado: ${selectedNode.id}`, selectedNode);

        // Highlight logic
        const mesh = intersects[0].object as THREE.Mesh;
        (mesh.material as THREE.MeshPhongMaterial).emissive.setHex(0xffd700); // Gold glow
        setTimeout(() => {
            (mesh.material as THREE.MeshPhongMaterial).emissive.setHex(0x111111);
        }, 1000);
      }
    };

    window.addEventListener('click', onMouseClick);

    const animate = () => {
      requestAnimationFrame(animate);
      nodeGroup.rotation.y += 0.005;
      edgeGroup.rotation.y += 0.005;
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      window.removeEventListener('click', onMouseClick);
      if (mountRef.current) mountRef.current.removeChild(renderer.domElement);
    };
  }, [nodes, edges]);

  return <div ref={mountRef} style={{ width: '100%', height: '100vh', overflow: 'hidden' }} />;
};
