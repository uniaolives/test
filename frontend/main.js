// O Olho de Arkhe - S9 Visualizer
const qDisplay = document.getElementById('q-display');
const sDisplay = document.getElementById('s-display');
const teDisplay = document.getElementById('te-display');
const pressureDisplay = document.getElementById('hmt-pressure');
const flowDisplay = document.getElementById('hmt-flow');
const stateDisplay = document.getElementById('state-display');
const statusEl = document.getElementById('status');
const vkEl = document.getElementById('vk');
const tkrEl = document.getElementById('t_kr');

const modeEl = document.createElement('div');
modeEl.id = 'hyperclaw_mode';
modeEl.className = 'stat';
document.getElementById('info').appendChild(modeEl);

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

// Unified WebSocket Reality Stream
const socket = new WebSocket(`ws://${window.location.hostname}:8000/ws/reality`);

socket.onopen = () => {
    statusEl.innerText = "Status: Connected";
};

socket.onmessage = (event) => {
    try {
        const data = JSON.parse(event.data);
        updateUI(data);
        updateVisuals(data);
    } catch (e) {
        console.error("Error parsing WebSocket data:", e);
    }
};

socket.onclose = () => {
    statusEl.innerText = "Status: Disconnected";
};

function updateUI(data) {
    // Basic metrics
    if (data.q_value !== undefined) qDisplay.innerText = data.q_value.toFixed(3);
    if (data.s_index !== undefined) sDisplay.innerText = data.s_index.toFixed(3);

    // Derived metrics
    if (data.p_ac !== undefined) pressureDisplay.innerText = data.p_ac.toFixed(3);
    if (data.te_coupling !== undefined) teDisplay.innerText = data.te_coupling.toFixed(3);

    // Hydraulic data
    if (data.hydraulic) {
        pressureDisplay.innerText = data.hydraulic.pressure.toFixed(3);
        flowDisplay.innerText = `${data.hydraulic.state} (v=${data.hydraulic.flow_rate.toFixed(2)})`;
    }

    // State
    if (data.state) stateDisplay.innerText = data.state;

    // VK and other info
    if (data.vk) {
        vkEl.innerText = `VK: B:${data.vk.bio.toFixed(2)} A:${data.vk.aff.toFixed(2)} S:${data.vk.soc.toFixed(2)} C:${data.vk.cog.toFixed(2)}`;
    }
    if (data.t_kr !== undefined) {
        tkrEl.innerText = `t_KR: ${data.t_kr}s`;
    }
    if (data.hyperclaw_mode) {
        modeEl.innerText = `Mode: ${data.hyperclaw_mode}`;
    }

    const color = (data.q_value || 0) > 0.9 ? '#ff0066' : (data.q_value || 0) > 0.7 ? '#ffff00' : '#00ffcc';
    qDisplay.style.color = color;
}

function updateVisuals(data) {
    const q = data.q_value || 0;
    const s = data.s_index || 0;

    let scale = 1 + (q * 0.8);
    if (data.te_coupling) scale += (data.te_coupling * 0.5);
    if (data.hydraulic) scale += (data.hydraulic.pressure * 0.2);

    sphere.scale.lerp(new THREE.Vector3(scale, scale, scale), 0.1);

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
    ring.scale.set(1 + s / 5, 1 + s / 5, 1);
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
