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

export class TorusVisualizer {
    private scene: THREE.Scene;
    private camera: THREE.PerspectiveCamera;
    private renderer: any; // Using any for WebGPURenderer as types might be tricky
    private torus: THREE.Mesh;
    private material: THREE.MeshStandardMaterial;
    private vk: KatharosVector;

    // WebGPU related
    private device: GPUDevice | null = null;
    private computePipeline: GPUComputePipeline | null = null;
    private vkBuffer: GPUBuffer | null = null;
    private refBuffer: GPUBuffer | null = null;
    private bindGroup: GPUBindGroup | null = null;
    private shaderCode: string = '';

    constructor(container: HTMLElement) {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);

        // Attempt to use WebGPURenderer
        try {
            this.renderer = new WebGPURenderer({ antialias: true, alpha: true });
        } catch (e) {
            console.warn("WebGPURenderer failed, falling back to WebGLRenderer", e);
            this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        }

        this.renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(this.renderer.domElement);

        const geometry = new THREE.TorusKnotGeometry(10, 3, 100, 16);
        this.material = new THREE.MeshStandardMaterial({
            color: 0x00ffcc,
            metalness: 0.7,
            roughness: 0.2,
            emissive: 0x003322,
            wireframe: true
        });
        this.torus = new THREE.Mesh(geometry, this.material);
        this.scene.add(this.torus);

        const light = new THREE.PointLight(0xffffff, 1, 100);
        light.position.set(20, 20, 20);
        this.scene.add(light);
        this.scene.add(new THREE.AmbientLight(0x404040));

        this.camera.position.z = 30;

        this.vk = { bio: 0.618, aff: 0.5, soc: 0.5, cog: 0.5, q_permeability: 1.0 };

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

    public async updateVK(newVk: Partial<KatharosVector>) {
        this.vk = { ...this.vk, ...newVk };

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

        const color = new THREE.Color().setHSL(this.vk.bio * 0.3 + 0.5, 0.8, 0.5);
        this.material.color.copy(color);
        this.material.emissive.setHSL(this.vk.soc * 0.3 + 0.5, 0.8, 0.2 * this.vk.aff);

        const scale = 0.8 + this.vk.q_permeability * 0.4;
        this.torus.scale.set(scale, scale, scale);
    }

    public animate() {
        requestAnimationFrame(() => this.animate());

        const rotationSpeed = 0.005 + (this.vk.cog * 0.02);
        this.torus.rotation.x += rotationSpeed;
        this.torus.rotation.y += rotationSpeed * 1.5;

        this.renderer.render(this.scene, this.camera);
    }

    public getVK() {
        return this.vk;
    }
}
