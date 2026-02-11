// qhttp/viz/static/js/optimized_main.js
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';

const ARKHE_COLORS = {
    0: 0x00ffff,  // CIEF → Cyan (Química)
    1: 0x00ff00,  // CEIF → Verde (Energia+Química)
    2: 0xff00ff,  // ICEF → Magenta (Informação)
    3: 0xffff00,  // IECF → Amarelo (Info+Energia)
    4: 0xff8800,  // ECIF → Laranja (Energia)
    5: 0x8800ff   // EICF → Roxo (Energia+Info)
};

class QuantumVisualizer {
    constructor() {
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.FogExp2(0x000510, 0.02);

        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(0, 20, 40);

        this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        document.body.appendChild(this.renderer.domElement);

        this.setupPostProcessing();
        this.setupControls();
        this.setupScene();
        this.setupObjectPools();

        this.nodes = new Map();
        this.links = new Map();
        this.particleSystems = new Map();

        this.lastFrameTime = performance.now();
        this.frameCount = 0;

        this.connectWebSocket();
        this.animate();
    }

    setupPostProcessing() {
        this.composer = new EffectComposer(this.renderer);
        this.composer.addPass(new RenderPass(this.scene, this.camera));

        const bloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            1.5, 0.4, 0.85
        );
        bloomPass.threshold = 0.1;
        bloomPass.strength = 2.0;
        bloomPass.radius = 0.5;
        this.composer.addPass(bloomPass);
    }

    setupControls() {
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.autoRotate = true;
        this.controls.autoRotateSpeed = 0.5;
        this.controls.maxDistance = 100;
        this.controls.minDistance = 10;
    }

    setupScene() {
        const gridHelper = new THREE.GridHelper(100, 50, 0x00ffff, 0x003333);
        this.scene.add(gridHelper);

        const particleGeo = new THREE.BufferGeometry();
        const particleCount = 1000;
        const positions = new Float32Array(particleCount * 3);

        for (let i = 0; i < particleCount * 3; i += 3) {
            positions[i] = (Math.random() - 0.5) * 100;
            positions[i + 1] = (Math.random() - 0.5) * 50;
            positions[i + 2] = (Math.random() - 0.5) * 100;
        }

        particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const particleMat = new THREE.PointsMaterial({
            color: 0x00ffff,
            size: 0.1,
            transparent: true,
            opacity: 0.6
        });

        this.ambientParticles = new THREE.Points(particleGeo, particleMat);
        this.scene.add(this.ambientParticles);
    }

    setupObjectPools() {
        this.linkPool = [];
        this.maxLinks = 100;

        for (let i = 0; i < this.maxLinks; i++) {
            const geometry = new THREE.BufferGeometry();
            const material = new THREE.LineBasicMaterial({
                color: 0xffffff,
                transparent: true,
                opacity: 0.8,
                blending: THREE.AdditiveBlending
            });
            const line = new THREE.Line(geometry, material);
            line.visible = false;
            this.scene.add(line);
            this.linkPool.push({ mesh: line, inUse: false });
        }

        this.nodeGeometry = new THREE.IcosahedronGeometry(2, 2);
        this.nodeMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ffff,
            wireframe: true,
            transparent: true,
            opacity: 0.3
        });
        this.coreMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ffff,
            transparent: true,
            opacity: 0.9
        });
    }

    getLinkFromPool() {
        const available = this.linkPool.find(l => !l.inUse);
        if (available) {
            available.inUse = true;
            available.mesh.visible = true;
            return available.mesh;
        }
        return null;
    }

    returnLinkToPool(mesh) {
        const poolItem = this.linkPool.find(l => l.mesh === mesh);
        if (poolItem) {
            poolItem.inUse = false;
            poolItem.mesh.visible = false;
        }
    }

    updateNode(id, data) {
        if (!this.nodes.has(id)) {
            this.createNode(id, data);
        }

        const node = this.nodes.get(id);
        const load = data.load || 0.5;
        const coherence = data.coherence || 0.99;

        const scale = 1 + Math.sin(performance.now() * 0.002 * (1 + load)) * 0.1;
        node.group.scale.setScalar(scale);

        // Arkhe dynamic colors update
        if (data.dominant_component !== undefined) {
             node.core.material.color.setHex(ARKHE_COLORS[data.dominant_component] || 0x00ffff);
             node.core.material.opacity = 0.5 + coherence * 0.5;
        } else {
            const hue = coherence * 0.33;
            node.core.material.color.setHSL(hue, 1, 0.5);
        }

        this.updateNodeParticles(id, data.agents || 0);
    }

    createNode(id, data) {
        const group = new THREE.Group();

        const mesh = new THREE.Mesh(this.nodeGeometry, this.nodeMaterial.clone());
        group.add(mesh);

        const core = new THREE.Mesh(
            new THREE.IcosahedronGeometry(1, 1),
            this.coreMaterial.clone()
        );
        group.add(core);

        const positions = {
            "q1": new THREE.Vector3(0, 5, -15),
            "q2": new THREE.Vector3(-13, -5, 10),
            "q3": new THREE.Vector3(13, -5, 10),
            "arkhe-node-1": new THREE.Vector3(0, 5, -15),
            "arkhe-node-2": new THREE.Vector3(-13, -5, 10),
            "arkhe-node-3": new THREE.Vector3(13, -5, 10)
        };
        group.position.copy(positions[id] || new THREE.Vector3(Math.random()*20-10, Math.random()*20-10, Math.random()*20-10));

        this.createLabel(group, id);

        this.scene.add(group);
        this.nodes.set(id, { group, mesh, core, particles: null });
    }

    createLabel(parent, text) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 256;
        canvas.height = 64;

        ctx.fillStyle = 'rgba(0, 0, 0, 0)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.font = 'bold 24px monospace';
        ctx.fillStyle = '#00ffcc';
        ctx.textAlign = 'center';
        ctx.fillText(text, 128, 40);

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMat = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(spriteMat);
        sprite.position.y = 3;
        sprite.scale.set(8, 2, 1);
        parent.add(sprite);
    }

    updateNodeParticles(nodeId, agentCount) {
        const node = this.nodes.get(nodeId);
        if (!node) return;

        if (node.particles) {
            this.scene.remove(node.particles);
        }

        const particleCount = Math.min(agentCount, 100);
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);

        for (let i = 0; i < particleCount; i++) {
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.random() * Math.PI;
            const r = 3 + Math.random() * 2;

            positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
            positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
            positions[i * 3 + 2] = r * Math.cos(phi);
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        const material = new THREE.PointsMaterial({
            color: 0xff00ff,
            size: 0.15,
            transparent: true,
            opacity: 0.8
        });

        const particles = new THREE.Points(geometry, material);
        particles.position.copy(node.group.position);
        node.particles = particles;
        this.scene.add(particles);
    }

    updateLinks(linksData) {
        const currentIds = new Set(linksData.map(l => l.id));
        const toRemove = [];

        this.links.forEach((link, id) => {
            if (!currentIds.has(id)) {
                toRemove.push(id);
            }
        });

        toRemove.forEach(id => {
            const link = this.links.get(id);
            this.returnLinkToPool(link.mesh);
            this.links.delete(id);
        });

        linksData.forEach(linkData => {
            if (this.links.has(linkData.id)) {
                this.updateLinkVisuals(this.links.get(linkData.id), linkData);
            } else {
                this.createLink(linkData);
            }
        });
    }

    createLink(data) {
        const mesh = this.getLinkFromPool();
        if (!mesh) {
            console.warn('Link pool exhausted');
            return;
        }

        this.updateLinkGeometry(mesh, data);
        this.links.set(data.id, { mesh, data });
    }

    updateLinkGeometry(mesh, data) {
        const nodeA = this.nodes.get(data.u);
        const nodeB = this.nodes.get(data.v);

        if (!nodeA || !nodeB) return;

        const p1 = nodeA.group.position;
        const p2 = nodeB.group.position;

        const fidelity = data.fidelity || 0.99;
        const mid = new THREE.Vector3().lerpVectors(p1, p2, 0.5);
        mid.y += 5 + (1 - fidelity) * 10;

        const curve = new THREE.QuadraticBezierCurve3(p1, mid, p2);
        const points = curve.getPoints(50);

        mesh.geometry.setFromPoints(points);
        mesh.material.color.setHex(this.getBellColor(data.state));
        mesh.material.opacity = 0.3 + fidelity * 0.7;
    }

    updateLinkVisuals(link, data) {
        const time = performance.now() * 0.001;
        link.mesh.material.opacity = 0.5 + Math.sin(time * 2) * 0.3;

        if (Math.abs(link.data.fidelity - data.fidelity) > 0.01) {
            this.updateLinkGeometry(link.mesh, data);
            link.data = data;
        }
    }

    getBellColor(state) {
        const colors = {
            0: 0x00ff00, // Φ+
            1: 0x0088ff, // Φ-
            2: 0xff00ff, // Ψ+
            3: 0xff4400  // Ψ-
        };
        return colors[state] || 0xffffff;
    }

    triggerCollapse(data) {
        const node = this.nodes.get(data.node);
        if (!node) return;

        const geometry = new THREE.RingGeometry(0.1, 0.5, 32);
        const material = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            side: THREE.DoubleSide
        });
        const shockwave = new THREE.Mesh(geometry, material);
        shockwave.position.copy(node.group.position);
        shockwave.lookAt(this.camera.position);
        this.scene.add(shockwave);

        const startTime = performance.now();
        const duration = 1000;

        const animate = () => {
            const elapsed = performance.now() - startTime;
            const progress = elapsed / duration;

            if (progress < 1) {
                shockwave.scale.setScalar(1 + progress * 20);
                material.opacity = 1 - progress;
                requestAnimationFrame(animate);
            } else {
                this.scene.remove(shockwave);
                geometry.dispose();
                material.dispose();
            }
        };
        animate();
    }

    handleArkheUpdate(event) {
        const { node, coherence, dominant_component } = event;
        const nodeObj = this.nodes.get(node);
        if (nodeObj) {
            nodeObj.core.material.color.setHex(ARKHE_COLORS[dominant_component]);
            nodeObj.core.material.opacity = 0.5 + coherence * 0.5;
        }
    }

    connectWebSocket() {
        const wsUrl = `ws://${window.location.host}/ws/quantum_stream`;
        this.ws = new WebSocket(wsUrl);

        this.ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);

            switch(msg.type) {
                case 'SNAPSHOT':
                    Object.entries(msg.nodes).forEach(([id, data]) => {
                        this.updateNode(id, data);
                    });
                    this.updateLinks(msg.links);
                    break;

                case 'EVENT':
                    if (msg.channel === 'arkhe:evolution') {
                        this.handleArkheUpdate(msg.data);
                    } else if (msg.data?.type === 'COLLAPSE') {
                        this.triggerCollapse(msg.data);
                    }
                    break;
            }
        };

        this.ws.onclose = () => {
            setTimeout(() => this.connectWebSocket(), 3000);
        };
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const now = performance.now();
        this.ambientParticles.rotation.y += 0.0005;

        this.nodes.forEach(node => {
            if (node.particles) {
                node.particles.rotation.y += 0.01;
                node.particles.rotation.x += 0.005;
            }
            node.group.rotation.y += 0.005;
        });

        this.controls.update();
        this.composer.render();

        this.frameCount++;
        if (now - this.lastFrameTime >= 1000) {
            this.frameCount = 0;
            this.lastFrameTime = now;
        }
    }
}

const viz = new QuantumVisualizer();
window.addEventListener('resize', () => {
    viz.camera.aspect = window.innerWidth / window.innerHeight;
    viz.camera.updateProjectionMatrix();
    viz.renderer.setSize(window.innerWidth, window.innerHeight);
    viz.composer.setSize(window.innerWidth, window.innerHeight);
});
