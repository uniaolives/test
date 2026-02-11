import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// --- SETUP DA CENA ---
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x000000, 0.02);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 20, 40);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);

// --- OBJETOS ---
const nodeMeshes = {};
const linkMeshes = {};

const nodeGeo = new THREE.IcosahedronGeometry(2, 1);
const nodeMat = new THREE.MeshBasicMaterial({ color: 0x00ffff, wireframe: true });

const nodePositions = {
    "arkhe-q1": new THREE.Vector3(-15, 0, 10),
    "arkhe-q2": new THREE.Vector3(15, 0, 10)
};

Object.keys(nodePositions).forEach(id => {
    const mesh = new THREE.Mesh(nodeGeo, nodeMat);
    mesh.position.copy(nodePositions[id]);
    scene.add(mesh);
    nodeMeshes[id] = mesh;
});

// --- WEBSOCKET ---
const ws = new WebSocket(`ws://${window.location.host}/ws/quantum_stream`);
const statusDiv = document.getElementById('status');
const metricsDiv = document.getElementById('metrics');

ws.onopen = () => { statusDiv.innerText = "🟢 SYSTEM ONLINE"; statusDiv.style.color = "#0f0"; };
ws.onclose = () => { statusDiv.innerText = "🔴 DISCONNECTED"; statusDiv.style.color = "#f00"; };

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "SNAPSHOT") {
        updateEntanglements(msg.entanglements);
    } else if (msg.type === "EVENT" && msg.event_type === "COLLAPSE") {
        triggerCollapseEffect(msg.data);
    }
};

function updateEntanglements(activePairs) {
    metricsDiv.innerText = `Active Pairs: ${activePairs.length}`;
    const currentIds = new Set(activePairs.map(p => p.pair_id));

    Object.keys(linkMeshes).forEach(id => {
        if (!currentIds.has(id)) {
            scene.remove(linkMeshes[id]);
            linkMeshes[id].geometry.dispose();
            delete linkMeshes[id];
        }
    });

    activePairs.forEach(pair => {
        if (!linkMeshes[pair.pair_id]) {
            createEntanglementLink(pair);
        }
    });
}

function createEntanglementLink(pair) {
    const posA = nodePositions[pair.node_a];
    const posB = nodePositions[pair.node_b];
    if (!posA || !posB) return;

    const midPoint = new THREE.Vector3().addVectors(posA, posB).multiplyScalar(0.5);
    midPoint.y += 10;

    const curve = new THREE.QuadraticBezierCurve3(posA, midPoint, posB);
    const points = curve.getPoints(50);
    const geometry = new THREE.BufferGeometry().setFromPoints(points);

    const colors = [0x00ff00, 0x0000ff, 0xff00ff, 0xff0000];
    const material = new THREE.LineBasicMaterial({
        color: colors[pair.bell_type] || 0xffffff,
        transparent: true,
        opacity: 0.8
    });

    const curveObject = new THREE.Line(geometry, material);
    scene.add(curveObject);
    linkMeshes[pair.pair_id] = curveObject;
}

function triggerCollapseEffect(data) {
    const pos = nodePositions[data.node];
    if (pos) {
        const geo = new THREE.SphereGeometry(1, 16, 16);
        const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true });
        const sphere = new THREE.Mesh(geo, mat);
        sphere.position.copy(pos);
        scene.add(sphere);

        const animateCollapse = () => {
            sphere.scale.multiplyScalar(1.1);
            sphere.material.opacity -= 0.05;
            if (sphere.material.opacity > 0) {
                requestAnimationFrame(animateCollapse);
            } else {
                scene.remove(sphere);
            }
        };
        animateCollapse();
    }
}

function animate() {
    requestAnimationFrame(animate);
    Object.values(nodeMeshes).forEach(mesh => {
        mesh.rotation.y += 0.005;
        mesh.rotation.x += 0.002;
    });
    controls.update();
    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
