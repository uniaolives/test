// O Olho de Arkhe - S9 Visualizer
const qDisplay = document.getElementById('q-display');
const sDisplay = document.getElementById('s-display');
const teDisplay = document.getElementById('te-display');
const pressureDisplay = document.getElementById('hmt-pressure');
const flowDisplay = document.getElementById('hmt-flow');
const stateDisplay = document.getElementById('state-display');

// Setup Three.js
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('container').appendChild(renderer.domElement);

// Arkhe Sphere
const geometry = new THREE.IcosahedronGeometry(2, 4);
const material = new THREE.MeshBasicMaterial({
    color: 0x00ffcc,
    wireframe: true,
    transparent: true,
    opacity: 0.5
});
const sphere = new THREE.Mesh(geometry, material);
scene.add(sphere);

// Synchronicity Ring
const ringGeo = new THREE.TorusGeometry(3, 0.05, 16, 100);
const ringMat = new THREE.MeshBasicMaterial({ color: 0xff0066 });
const ring = new THREE.Mesh(ringGeo, ringMat);
scene.add(ring);

camera.position.z = 7;

// WebSocket Reality Stream
const socket = new WebSocket(`ws://${window.location.hostname}:8000/ws/reality`);

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateUI(data);
    updateVisuals(data);
};

function updateUI(data) {
    qDisplay.innerText = data.q_value.toFixed(3);
    sDisplay.innerText = data.s_index.toFixed(3);
    if (data.te_coupling) teDisplay.innerText = data.te_coupling.toFixed(3);
    if (data.hydraulic) {
        pressureDisplay.innerText = data.hydraulic.pressure.toFixed(3);
        flowDisplay.innerText = `${data.hydraulic.state} (v=${data.hydraulic.flow_rate.toFixed(2)})`;
    }
    stateDisplay.innerText = data.state;

    const color = data.q_value > 0.9 ? '#ff0066' : data.q_value > 0.7 ? '#ffff00' : '#00ffcc';
    qDisplay.style.color = color;
}

function updateVisuals(data) {
    let scale = 1 + (data.q_value * 0.8);
    if (data.te_coupling) scale += (data.te_coupling * 0.5);
    if (data.hydraulic) scale += (data.hydraulic.pressure * 0.2);

    sphere.scale.lerp(new THREE.Vector3(scale, scale, scale), 0.1);

    const q = data.q_value;
    if (q > 0.95) {
        material.color.setHex(0xff0066);
        material.opacity = 1.0;
    } else if (q > 0.7) {
        material.color.setHex(0xffff00);
        material.opacity = 0.8;
    } else {
        material.color.setHex(0x00ffcc);
        material.opacity = 0.5;
    }

    ring.rotation.z += 0.01 * (1 + q * 10);
    ring.scale.set(1 + data.s_index / 5, 1 + data.s_index / 5, 1);
}

function animate() {
    requestAnimationFrame(animate);
    sphere.rotation.x += 0.002;
    sphere.rotation.y += 0.003;
    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
