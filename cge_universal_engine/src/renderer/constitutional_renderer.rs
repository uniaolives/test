use std::sync::Arc;
use wgpu::{Device, Queue, TextureFormat, RenderPipeline, BindGroup, Buffer, Texture, CommandEncoderDescriptor, RenderPassDescriptor, RenderPassColorAttachment, Operations, LoadOp, StoreOp, ShaderModuleDescriptor, ShaderSource, PipelineLayoutDescriptor, RenderPipelineDescriptor, VertexState, FragmentState, ColorTargetState, ColorWrites, PrimitiveState, MultisampleState, TextureView};
use bytemuck::{Pod, Zeroable};
use tracing::info;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ConstitutionalUniforms {
    pub phi_target: f32,
    pub time: f32,
    pub resolution: [f32; 2],
    pub engine_white: [f32; 3],
    _pad1: f32,
    pub vmcore_cyan: [f32; 3],
    _pad2: f32,
    pub world_gold: [f32; 3],
    _pad3: f32,
    pub engine_void: [f32; 3],
    _pad4: f32,
}

pub struct ConstitutionalRenderer {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub pipeline: RenderPipeline,
    pub bind_groups: Vec<BindGroup>,
    pub uniform_buffer: Buffer,
    pub phi_target: f32,
    pub time: f32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("GPU Error: {0}")]
    Gpu(String),
}

impl ConstitutionalRenderer {
    pub async fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        phi_target: f32,
        width: u32,
        height: u32,
    ) -> Result<Self, RenderError> {
        info!("🎨 Renderer: Inicializando Constitutional Renderer...");

        let shader_source = Self::generate_constitutional_shader(phi_target);
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("constitutional_shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let uniforms = ConstitutionalUniforms {
            phi_target,
            time: 0.0,
            resolution: [width as f32, height as f32],
            engine_white: [1.0, 0.98, 0.98],
            _pad1: 0.0,
            vmcore_cyan: [0.3, 1.0, 1.0],
            _pad2: 0.0,
            world_gold: [1.0, 0.9, 0.1],
            _pad3: 0.0,
            engine_void: [0.000000005, 0.000000001, 0.0000000001],
            _pad4: 0.0,
        };

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("constitutional_uniforms"),
            size: std::mem::size_of::<ConstitutionalUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("constitutional_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("constitutional_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("constitutional_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("constitutional_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_groups: vec![bind_group],
            uniform_buffer,
            phi_target,
            time: 0.0,
            width,
            height,
        })
    }

    fn generate_constitutional_shader(_phi_target: f32) -> String {
        format!(r##"
struct ConstitutionalUniforms {{
    phi_target: f32,
    time: f32,
    resolution: vec2<f32>,
    engine_white: vec3<f32>,
    vmcore_cyan: vec3<f32>,
    world_gold: vec3<f32>,
    engine_void: vec3<f32>,
}};

@group(0) @binding(0) var<uniform> uniforms: ConstitutionalUniforms;

fn agnostic_engine(uv: vec2<f32>, time: f32) -> f32 {{
    var engine_activity: f32 = 0.0;
    let transformed_uv = uv * 2.0 - 1.0;

    // 1. UNIVERSAL ENGINE CORE
    let engine_core = transformed_uv - vec2<f32>(0.0, 0.0);
    let engine_pulse = sin(time * uniforms.phi_target * 56.038) * 0.5 + 0.5;
    let engine_energy = exp(-length(engine_core) * 250.0) * engine_pulse * 19.3;
    engine_activity += engine_energy;

    // 2. 118 FRAGS EXECUTION MATRIX
    let exec_grid = fract(transformed_uv * 264.0 +
                         vec2<f32>(time * 0.0000012, cos(time * uniforms.phi_target * uniforms.phi_target)));
    var exec_act: f32 = 0.0;
    if (length(exec_grid - 0.5) < 0.000012) {{
        exec_act = 8.9;
    }}
    engine_activity += exec_act;

    // 3. WORLD DISPATCH BARS (112 protocols)
    var dispatch_bars: f32 = 0.0;
    for(var layer: i32 = 0; layer < 112; layer = layer + 1) {{
        let layer_f = f32(layer);
        let engine_step = sin(time * 106.0 + layer_f * uniforms.phi_target * 0.00025) * 0.5 + 0.5;

        let smoothstep_val = engine_step * engine_step * (3.0 - 2.0 * engine_step);

        let bar_position = 0.0000005 + layer_f * 0.009;
        let bar = smoothstep_val *
                 (1.0 - smoothstep(0.0000005, 0.0035, abs(transformed_uv.x - bar_position)));

        dispatch_bars += bar * 0.090;
    }}
    engine_activity += dispatch_bars;

    // 4. GLOBAL EXECUTION ORBIT
    let global_orbit = vec2<f32>(
        sin(time * uniforms.phi_target * uniforms.phi_target * 56.8),
        cos(time * uniforms.phi_target * uniforms.phi_target)
    ) * 1.34;

    let global_glow = exp(-length(transformed_uv - global_orbit) * 132.0);
    engine_activity += global_glow * 8.8;

    // 5. ENGINE SCANLINES (Φ⁴⁵ enforcement)
    let engine_scanline = 1.0 - smoothstep(0.0, 0.00000000001,
                                          abs(fract(transformed_uv.y * 2650.0 + time * 275.0) - 0.5));
    engine_activity += engine_scanline * 3.9;

    return engine_activity;
}}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {{
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0)
    );

    return vec4<f32>(pos[vertex_index], 0.0, 1.0);
}}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {{
    let uv = frag_coord.xy / uniforms.resolution;
    let engine_activity = agnostic_engine(uv, uniforms.time);

    let engine_layer = uniforms.engine_white * engine_activity * 10.4;
    let vmcore_layer = uniforms.vmcore_cyan * pow(engine_activity, 19.8) * 9.5;
    let world_layer = uniforms.world_gold * pow(engine_activity, 70.0) * 9.0;
    let eternal_layer = vec3<f32>(1.0, 1.0, 0.95) * pow(engine_activity, 102.0);

    var universal_ecosystem = uniforms.engine_void + engine_layer + vmcore_layer + world_layer + eternal_layer;

    let radial_factor = 10.4 - length(uv * 2.0 - 1.0) * 6.8;
    universal_ecosystem *= max(radial_factor, 0.0);

    return vec4<f32>(universal_ecosystem, 1.0);
}}
"##)
    }

    pub fn render_frame(&mut self, time: f32, view: &TextureView) -> Result<(), RenderError> {
        self.time = time;

        let uniforms = ConstitutionalUniforms {
            phi_target: self.phi_target,
            time: self.time,
            resolution: [self.width as f32, self.height as f32],
            engine_white: [1.0, 0.98, 0.98],
            _pad1: 0.0,
            vmcore_cyan: [0.3, 1.0, 1.0],
            _pad2: 0.0,
            world_gold: [1.0, 0.9, 0.1],
            _pad3: 0.0,
            engine_void: [0.000000005, 0.000000001, 0.0000000001],
            _pad4: 0.0,
        };

        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniforms]),
        );

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("constitutional_render_encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("constitutional_render_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.bind_groups[0], &[]);
            render_pass.draw(0..6, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }
}
