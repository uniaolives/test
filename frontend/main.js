// O Olho de Arkhe - S9 Visualizer
const qDisplay = document.getElementById('q-display');
const sDisplay = document.getElementById('s-display');
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
    stateDisplay.innerText = data.state;

    const color = data.q_value > 0.9 ? '#ff0066' : data.q_value > 0.7 ? '#ffff00' : '#00ffcc';
    qDisplay.style.color = color;
}

function updateVisuals(data) {
    const scale = 1 + (data.q_value * 0.8);
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
const statusEl = document.getElementById('status');
const vkEl = document.getElementById('vk');
const tkrEl = document.getElementById('t_kr');
const modeEl = document.createElement('div');
modeEl.id = 'hyperclaw_mode';
modeEl.className = 'stat';
document.getElementById('info').appendChild(modeEl);

const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws/entrainment`);

ws.onopen = () => {
    statusEl.innerText = "Status: Metabolizing";
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    vkEl.innerText = `VK: B:${data.vk.bio.toFixed(2)} A:${data.vk.aff.toFixed(2)} S:${data.vk.soc.toFixed(2)} C:${data.vk.cog.toFixed(2)}`;
    tkrEl.innerText = `t_KR: ${data.t_kr}s`;
    modeEl.innerText = `Mode: ${data.hyperclaw_mode || 'N/A'}`;
};

ws.onclose = () => {
    statusEl.innerText = "Status: Disconnected";
};
