import * as THREE from 'three';
// @ts-ignore
import WebGPURenderer from 'three/addons/renderers/webgpu/WebGPURenderer.js';

export interface KatharosVector {
    bio: number;
    aff: number;
    soc: number;
    cog: number;
    q_permeability: number;
}

export interface NodeState {
    Q: number;
    deltaK: number;
    t_KR: number;
    isCrisis: boolean;
}

export class ArkheVisualizer {
    private scene: THREE.Scene;
    private camera: THREE.PerspectiveCamera;
    private renderer: any;
    private cube: THREE.Mesh;
    private sphere: THREE.Mesh;
    private agents: THREE.Mesh[] = [];
    private threads: THREE.Line[] = [];
    private cubeMaterial: THREE.MeshStandardMaterial;
    private sphereMaterial: THREE.MeshPhysicalMaterial;
    private vk: KatharosVector;
    private state: NodeState;

    // WebGPU related
    private device: GPUDevice | null = null;
    private computePipeline: GPUComputePipeline | null = null;
    private vkBuffer: GPUBuffer | null = null;
    private refBuffer: GPUBuffer | null = null;
    private bindGroup: GPUBindGroup | null = null;
    private shaderCode: string = '';

    constructor(container: HTMLElement) {
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.FogExp2(0x020202, 0.05);
        this.camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);

        try {
            this.renderer = new WebGPURenderer({ antialias: true, alpha: true });
        } catch (e) {
            console.warn("WebGPURenderer failed, falling back to WebGLRenderer", e);
            this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        }

        this.renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(this.renderer.domElement);

        // Core Cube
        const cubeGeo = new THREE.BoxGeometry(1.5, 1.5, 1.5);
        this.cubeMaterial = new THREE.MeshStandardMaterial({
            color: 0x111111, metalness: 0.9, roughness: 0.1, emissive: 0x002222
        });
        this.cube = new THREE.Mesh(cubeGeo, this.cubeMaterial);
        this.scene.add(this.cube);

        // Membrane Sphere
        const sphereGeo = new THREE.SphereGeometry(2.5, 32, 32);
        this.sphereMaterial = new THREE.MeshPhysicalMaterial({
            color: 0x00ffcc, transmission: 0.9, opacity: 1, transparent: true, roughness: 0.0, clearcoat: 1.0
        });
        this.sphere = new THREE.Mesh(sphereGeo, this.sphereMaterial);
        this.scene.add(this.sphere);

        // Agents & Mycelial Threads
        const agentGeo = new THREE.SphereGeometry(0.15, 16, 16);
        const agentMat = new THREE.MeshBasicMaterial({ color: 0xaaffaa });
        const lineMat = new THREE.LineBasicMaterial({ color: 0x00ffcc, transparent: true, opacity: 0.5 });

        for (let i = 0; i < 12; i++) {
            const agent = new THREE.Mesh(agentGeo, agentMat.clone());
            const angle = (i / 12) * Math.PI * 2;
            agent.userData = {
                angle: angle,
                radius: 4 + Math.random() * 2,
                speed: 0.005 + Math.random() * 0.01,
                reputation: 50 + Math.random() * 100
            };
            this.scene.add(agent);
            this.agents.push(agent);

            const lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(0,0,0)]);
            const thread = new THREE.Line(lineGeo, lineMat.clone());
            this.scene.add(thread);
            this.threads.push(thread);
        }

        const ambientLight = new THREE.AmbientLight(0x223333);
        this.scene.add(ambientLight);
        const coreLight = new THREE.PointLight(0x00ffcc, 1.5, 20);
        this.scene.add(coreLight);

        this.camera.position.z = 8;

        this.vk = { bio: 0.618, aff: 0.5, soc: 0.5, cog: 0.5, q_permeability: 1.0 };
        this.state = { Q: 1.0, deltaK: 0.05, t_KR: 100, isCrisis: false };

        window.addEventListener('resize', () => this.onWindowResize(container));
    }

    public async initializeWebGPU(shaderCode: string) {
        if (!navigator.gpu) {
            console.error("WebGPU not supported on this browser.");
            return;
        }

        this.device = await (await navigator.gpu.requestAdapter())?.requestDevice() || null;
        if (!this.device) return;

        this.shaderCode = shaderCode;

        const shaderModule = this.device.createShaderModule({
            code: this.shaderCode,
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });

        this.vkBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        this.refBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.bindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.vkBuffer } },
                { binding: 1, resource: { buffer: this.refBuffer } },
            ],
        });

        const refData = new Float32Array([0.618, 0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0]);
        this.device.queue.writeBuffer(this.refBuffer, 0, refData);
    }

    private onWindowResize(container: HTMLElement) {
        this.camera.aspect = container.clientWidth / container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(container.clientWidth, container.clientHeight);
    }

    public async updateState(newState: Partial<NodeState>, vk: Partial<KatharosVector>) {
        this.state = { ...this.state, ...newState };
        this.vk = { ...this.vk, ...vk };

        if (this.device && this.computePipeline && this.vkBuffer && this.bindGroup) {
            const vkData = new Float32Array([
                this.vk.bio, this.vk.aff, this.vk.soc, this.vk.cog,
                this.vk.q_permeability, 0.0, 0.0, 0.0
            ]);
            this.device.queue.writeBuffer(this.vkBuffer, 0, vkData);

            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.computePipeline);
            passEncoder.setBindGroup(0, this.bindGroup);
            passEncoder.dispatchWorkgroups(1);
            passEncoder.end();

            const readBuffer = this.device.createBuffer({
                size: 32,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            commandEncoder.copyBufferToBuffer(this.vkBuffer, 0, readBuffer, 0, 32);

            this.device.queue.submit([commandEncoder.finish()]);

            await readBuffer.mapAsync(GPUMapMode.READ);
            const result = new Float32Array(readBuffer.getMappedRange());
            this.vk.q_permeability = result[4];
            readBuffer.unmap();
            readBuffer.destroy();
        }

        // Apply Visual Effects
        this.sphereMaterial.transmission = this.state.Q;
        this.sphereMaterial.color.setHex(this.state.isCrisis ? 0xff3366 : 0x00ffcc);
        const targetRadius = this.state.isCrisis ? 0.8 : 1.0;
        this.sphere.scale.lerp(new THREE.Vector3(targetRadius, targetRadius, targetRadius), 0.05);

        const stressPulse = this.state.isCrisis ? Math.sin(Date.now() * 0.01) * 0.2 : 0;
        this.cube.scale.set(1 + stressPulse, 1 + stressPulse, 1 + stressPulse);
        this.cubeMaterial.emissive.setHex(this.state.isCrisis ? 0x440000 : 0x002222);

        this.agents.forEach((agent, i) => {
            agent.userData.angle += agent.userData.speed * (this.state.isCrisis ? 3 : 1);
            const x = Math.cos(agent.userData.angle) * agent.userData.radius;
            const z = Math.sin(agent.userData.angle) * agent.userData.radius;
            const y = Math.sin(agent.userData.angle * 2) * 1.5;
            agent.position.set(x, y, z);

            const isMatureEnough = agent.userData.reputation > 60;
            let Q_ij = 0;
            if (!this.state.isCrisis && isMatureEnough) {
                Q_ij = this.state.Q;
            }

            const thread = this.threads[i];
            if (Q_ij > 0.3) {
                const positions = new Float32Array([
                    this.cube.position.x, this.cube.position.y, this.cube.position.z,
                    agent.position.x, agent.position.y, agent.position.z
                ]);
                thread.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                thread.material.opacity = Q_ij * 0.6;
                thread.visible = true;
                (agent.material as THREE.MeshBasicMaterial).color.setHex(0x00ffcc);
            } else {
                thread.visible = false;
                (agent.material as THREE.MeshBasicMaterial).color.setHex(0x555555);
            }
        });
    }

    public animate() {
        requestAnimationFrame(() => this.animate());

        this.cube.rotation.x += 0.005;
        this.cube.rotation.y += 0.008;
        this.sphere.rotation.y -= 0.002;
        this.sphere.rotation.z -= 0.001;

        this.renderer.render(this.scene, this.camera);
    }

    public getVK() {
        return this.vk;
    }

    public getState() {
        return this.state;
    }
}
