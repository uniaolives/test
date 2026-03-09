// Arkhe(n) Dashboard - Three.js Visualization
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });

renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('canvas-container').appendChild(renderer.domElement);

// Create the Quaternion Orb (Manifold Visualization)
const geometry = new THREE.SphereGeometry(1, 32, 32);
const material = new THREE.MeshBasicMaterial({ color: 0x9977ff, wireframe: true });
const orb = new THREE.Mesh(geometry, material);
scene.add(orb);

camera.position.z = 5;

// Simulate data updates
let phase = 0;
function animate() {
    requestAnimationFrame(animate);

    // Rotate orb based on phase
    orb.rotation.x += 0.01;
    orb.rotation.y += 0.01;

    phase = (phase + 0.02) % (2 * Math.PI);
    document.getElementById('phase').innerText = phase.toFixed(2);
    document.getElementById('coherence').innerText = (0.8 + 0.1 * Math.sin(Date.now() / 1000)).toFixed(2);

    renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

animate();
console.log("Arkhe(n) Kernel Dashboard Initialized.");
