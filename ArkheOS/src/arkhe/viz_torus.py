"""
Arkhe(n) 3D Torus Visualization
Maps hypergraph nodes onto a toroidal surface (S¹ × S¹).
"""

import json

def generate_torus_map(nodes, output_path="torus_map.html"):
    """
    Produces a standalone HTML file using Three.js to render nodes on a torus.
    """
    nodes_data = json.dumps(nodes)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Arkhe(N) Torus Map</title>
        <style>body {{ margin: 0; overflow: hidden; }} canvas {{ display: block; }}</style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body>
        <script>
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer();
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.body.appendChild(renderer.domElement);

            const nodes = {nodes_data};

            // Torus parameters
            const R = 5; // Major radius
            const r = 2; // Minor radius

            // Geometry
            const torusGeo = new THREE.TorusGeometry(R, r, 32, 100);
            const torusMat = new THREE.MeshBasicMaterial({{ color: 0x333333, wireframe: true, transparent: true, opacity: 0.1 }});
            const torus = new THREE.Mesh(torusGeo, torusMat);
            scene.add(torus);

            // Add nodes
            nodes.forEach(n => {{
                // Map omega (0..0.33) to theta, and some other coordinate to phi
                const theta = n.omega * Math.PI * 6; // major angle
                const phi = (n.id_num % 7) / 7 * Math.PI * 2; // minor angle

                const x = (R + r * Math.cos(phi)) * Math.cos(theta);
                const y = (R + r * Math.cos(phi)) * Math.sin(theta);
                const z = r * Math.sin(phi);

                const dotGeo = new THREE.SphereGeometry(0.1, 8, 8);
                const dotMat = new THREE.MeshBasicMaterial({{ color: n.coherence > 0.9 ? 0x00ff00 : 0x0000ff }});
                const dot = new THREE.Mesh(dotGeo, dotMat);
                dot.position.set(x, y, z);
                scene.add(dot);
            }});

            camera.position.z = 15;

            function animate() {{
                requestAnimationFrame(animate);
                torus.rotation.x += 0.005;
                torus.rotation.y += 0.005;
                renderer.render(scene, camera);
            }}
            animate();
        </script>
    </body>
    </html>
    """
    with open(output_path, "w") as f:
        f.write(html)
    print(f"🔮 Mapa do Toro gerado em {output_path}")

if __name__ == "__main__":
    # Sample nodes
    sample_nodes = [
        {{"id_num": 1, "omega": 0.0, "coherence": 0.86}},
        {{"id_num": 2, "omega": 0.07, "coherence": 0.94}},
        {{"id_num": 3, "omega": 0.33, "coherence": 0.81}}
    ]
    generate_torus_map(sample_nodes)
