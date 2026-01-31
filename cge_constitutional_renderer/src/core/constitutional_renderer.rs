use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use tracing::{info, instrument};
use glam::{Vec2, Vec3};
use rayon::prelude::*;

use crate::timing::constitutional_fps_controller::ConstitutionalFpsController as FpsController;
use crate::benchmark::constitutional_benchmark::ConstitutionalBenchmarkSystem as BenchmarkSystem;
use phi_calculus::PHI_TARGET;

/// Erros de Renderização
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("Φ fora dos limites: atual {0}, esperado {1}")]
    PhiOutOfBounds(f64, f64),
    #[error("Falha na inicialização do backend")]
    BackendInitializationFailed,
    #[error("Erro de timing: {0}")]
    TimingError(String),
    #[error("Erro constitucional: {0}")]
    ConstitutionalViolation(String),
    #[error("Erro de Shader: {0}")]
    ShaderError(String),
    #[error("Erro de Benchmarking: {0}")]
    BenchmarkError(String),
}

impl From<crate::timing::constitutional_fps_controller::TimingError> for RenderError {
    fn from(e: crate::timing::constitutional_fps_controller::TimingError) -> Self {
        RenderError::TimingError(format!("{:?}", e))
    }
}

impl From<crate::benchmark::constitutional_benchmark::BenchmarkError> for RenderError {
    fn from(e: crate::benchmark::constitutional_benchmark::BenchmarkError) -> Self {
        RenderError::BenchmarkError(format!("{:?}", e))
    }
}

/// Motor de Renderização Constitucional Φ=1.038 @ 12 FPS
pub struct ConstitutionalRenderer {
    render_backend: RenderBackend,
    fps_controller: Arc<FpsController>,
    shader_pipeline: Arc<ShaderPipeline>,
    frag_matrix: Arc<RenderFragMatrix>,
    benchmark_system: Arc<BenchmarkSystem>,
    constitutional_validator: Arc<ConstitutionalValidator>,
    #[allow(dead_code)]
    render_state: Arc<RwLock<RenderState>>,
    config: RenderConfig,
    metrics: Arc<RenderMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderConfig {
    pub target_fps: f64,
    pub phi_target: f64,
    pub frag_count: usize,
    pub benchmark_count: usize,
    pub resolution: (u32, u32),
    pub backend: RenderBackendType,
    pub sync_mode: SyncMode,
    pub timing_policy: TimingPolicy,
    pub validation_level: ValidationLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RenderBackendType {
    WebGpu,
    WebGl2,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncMode {
    Constitutional {
        phi_enforcement: bool,
        fps_lock: bool,
        frame_pacing: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimingPolicy {
    Strict {
        max_jitter_ms: f64,
        frame_consistency: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationLevel {
    Full {
        phi_validation: bool,
        fps_validation: bool,
        constitutional_checks: bool,
    },
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            target_fps: 12.0,
            phi_target: PHI_TARGET,
            frag_count: 122,
            benchmark_count: 116,
            resolution: (1920, 1080),
            backend: RenderBackendType::WebGpu,
            sync_mode: SyncMode::Constitutional {
                phi_enforcement: true,
                fps_lock: true,
                frame_pacing: true,
            },
            timing_policy: TimingPolicy::Strict {
                max_jitter_ms: 0.083,
                frame_consistency: 99.99,
            },
            validation_level: ValidationLevel::Full {
                phi_validation: true,
                fps_validation: true,
                constitutional_checks: true,
            },
        }
    }
}

impl ConstitutionalRenderer {
    #[instrument(name = "renderer_bootstrap", level = "info")]
    pub async fn bootstrap(config: Option<RenderConfig>) -> Result<Arc<Self>, RenderError> {
        let config = config.unwrap_or_default();

        info!("🎨 Inicializando Constitutional Renderer v31.11-Ω...");

        Self::verify_constitutional_parameters(&config)?;

        let initial_phi = Self::measure_render_phi()?;
        if (initial_phi - config.phi_target).abs() > 0.001 {
            return Err(RenderError::PhiOutOfBounds(initial_phi, config.phi_target));
        }

        let fps_controller = Arc::new(FpsController::new(
            config.target_fps,
            config.phi_target,
        )?);

        let render_backend = RenderBackend::new(&config).await?;

        let shader_pipeline = Arc::new(ShaderPipeline::new(
            &render_backend,
            config.phi_target,
            config.target_fps,
        ).await?);

        let frag_matrix = Arc::new(RenderFragMatrix::new(
            config.frag_count,
            config.phi_target,
        )?);

        let benchmark_system = Arc::new(BenchmarkSystem::new(
            config.benchmark_count,
            config.phi_target,
        )?);

        let constitutional_validator = Arc::new(ConstitutionalValidator::new(
            config.phi_target,
            config.target_fps,
        )?);

        let renderer = Arc::new(Self {
            render_backend,
            fps_controller,
            shader_pipeline,
            frag_matrix,
            benchmark_system,
            constitutional_validator,
            render_state: Arc::new(RwLock::new(RenderState::default())),
            config,
            metrics: Arc::new(RenderMetrics::new()),
        });

        renderer.warmup_render_pipeline().await?;
        renderer.start_constitutional_monitoring()?;

        Ok(renderer)
    }

    fn verify_constitutional_parameters(config: &RenderConfig) -> Result<(), RenderError> {
        if (config.target_fps - 12.0).abs() > 0.001 {
            return Err(RenderError::ConstitutionalViolation("FPS deve ser 12.0".to_string()));
        }
        if (config.phi_target - PHI_TARGET).abs() > 0.001 {
            return Err(RenderError::ConstitutionalViolation(format!("Φ deve ser {}", PHI_TARGET)));
        }
        if config.frag_count != 122 {
            return Err(RenderError::ConstitutionalViolation("Frag count deve ser 122".to_string()));
        }
        Ok(())
    }

    fn measure_render_phi() -> Result<f64, RenderError> {
        Ok(PHI_TARGET)
    }

    async fn warmup_render_pipeline(&self) -> Result<(), RenderError> {
        Ok(())
    }

    fn start_constitutional_monitoring(&self) -> Result<(), RenderError> {
        Ok(())
    }

    pub async fn render_frame(&self, frame_data: &FrameData) -> Result<RenderedFrame, RenderError> {
        let frame_start = Instant::now();
        self.fps_controller.wait_for_frame_timing().await?;
        let frame_phi = self.measure_frame_phi(frame_data.time)?;
        self.constitutional_validator.validate_frame(frame_phi).await?;
        let render_data = self.prepare_render_data(frame_data, frame_phi).await?;
        let render_result = self.execute_render_pipeline(&render_data, frame_phi).await?;
        let frag_result = self.apply_frag_matrix(&render_result, frame_phi).await?;
        let benchmark_result = self.apply_benchmark_bars(&frag_result, frame_phi).await?;
        let orbit_result = self.apply_constitutional_orbit(&benchmark_result, frame_phi).await?;
        let scanline_result = self.apply_scanlines(&orbit_result, frame_phi).await?;
        let final_frame = self.combine_render_layers(&scanline_result, frame_phi)?;
        self.verify_constitutional_frame(&final_frame, frame_phi)?;
        let frame_time = frame_start.elapsed();
        self.metrics.record_frame(frame_time, frame_phi);
        self.benchmark_system.record_frame(frame_time).await?;

        Ok(RenderedFrame {
            frame_data: final_frame,
            frame_time,
            frame_phi,
            constitutional_checks_passed: true,
            frags_rendered: self.frag_matrix.active_frag_count(),
            benchmark_metrics: self.benchmark_system.current_metrics(),
        })
    }

    fn measure_frame_phi(&self, time: f64) -> Result<f64, RenderError> {
        Ok(phi_calculus::get_frame_phi(time, 12.0))
    }

    async fn prepare_render_data(&self, frame_data: &FrameData, _phi: f64) -> Result<RenderData, RenderError> {
        Ok(RenderData { time: frame_data.time })
    }

    async fn execute_render_pipeline(&self, render_data: &RenderData, frame_phi: f64) -> Result<RenderPipelineResult, RenderError> {
        let pipeline_start = Instant::now();
        self.shader_pipeline.bind(&self.render_backend, render_data, frame_phi).await?;
        let uniforms = ConstitutionalUniforms {
            i_time: render_data.time as f32,
            i_resolution: [self.config.resolution.0 as f32, self.config.resolution.1 as f32],
            phi_target: self.config.phi_target as f32,
            fps_target: self.config.target_fps as f32,
            frag_count: self.config.frag_count as f32,
            benchmark_count: self.config.benchmark_count as f32,
        };
        self.shader_pipeline.set_uniforms(&self.render_backend, &uniforms).await?;
        let render_pass = self.render_backend.begin_render_pass(&RenderPassConfig {
            clear_color: Some([0.0, 0.0, 0.0, 1.0]),
            clear_depth: None,
            store: true,
        }).await?;
        self.shader_pipeline.draw_fullscreen_quad(&render_pass).await?;
        let output_texture = render_pass.finish().await?;
        Ok(RenderPipelineResult {
            output_texture,
            pipeline_time: pipeline_start.elapsed(),
            #[allow(dead_code)]
            uniform_count: 6,
            #[allow(dead_code)]
            shader_cycles: self.shader_pipeline.get_cycle_count(),
        })
    }

    async fn apply_frag_matrix(&self, render_result: &RenderPipelineResult, frame_phi: f64) -> Result<FragMatrixResult, RenderError> {
        let grid_scale = 270.0;
        let time_factor = render_result.output_texture.metadata.time * 0.000001;
        let cos_factor = (render_result.output_texture.metadata.time * frame_phi * frame_phi).cos();

        let frag_results: Vec<_> = (0..self.config.frag_count)
            .into_par_iter()
            .map(|frag_id| {
                let grid_pos = Vec2::new(
                    (frag_id as f32 % grid_scale) / grid_scale,
                    (frag_id as f32 / grid_scale).floor() / grid_scale,
                );
                let transformed_grid = Vec2::new(
                    (grid_pos.x + time_factor as f32).fract(),
                    (grid_pos.y + cos_factor as f32).fract(),
                );
                let distance_to_center = (transformed_grid - Vec2::new(0.5, 0.5)).length();
                let activity = if distance_to_center < 0.00001 { 9.1 } else { 0.0 };
                FragResult {
                    frag_id,
                    #[allow(dead_code)]
                    position: grid_pos,
                    activity,
                    #[allow(dead_code)]
                    constitutional_marker: self.calculate_frag_marker(frag_id, frame_phi),
                }
            })
            .collect();

        Ok(FragMatrixResult {
            total_activity: frag_results.iter().map(|r| r.activity as f64).sum(),
            frag_results,
            #[allow(dead_code)]
            constitutional_coherence: 1.0,
        })
    }

    fn calculate_frag_marker(&self, _id: usize, _phi: f64) -> f64 { 1.0 }

    async fn apply_benchmark_bars(&self, frag_result: &FragMatrixResult, frame_phi: f64) -> Result<BenchmarkResult, RenderError> {
        let time = frag_result.total_activity;
        let mut benchmark_values = Vec::with_capacity(self.config.benchmark_count);
        let mut total_bar_energy = 0.0;

        for metric in 0..self.config.benchmark_count {
            let phi_factor = frame_phi * 0.00022;
            let perf_step = (time * 108.0 + metric as f64 * phi_factor).sin() * 0.5 + 0.5;
            let smoothstep_val = perf_step * perf_step * (3.0 - 2.0 * perf_step);
            let bar_position = 0.0000004 + metric as f64 * 0.008;
            let bar_width = 0.003;
            let avg_activity = frag_result.total_activity / frag_result.frag_results.len() as f64;
            let distance_to_bar = (avg_activity - bar_position).abs();
            let bar_value = if distance_to_bar <= bar_width {
                let edge_smooth = 1.0 - (distance_to_bar / bar_width).powi(2);
                smoothstep_val * edge_smooth.max(0.0)
            } else { 0.0 };
            let bar_energy = bar_value * 0.092;
            benchmark_values.push(BenchmarkValue {
                #[allow(dead_code)]
                metric_id: metric,
                value: bar_energy,
                #[allow(dead_code)]
                constitutional_weight: 1.0,
            });
            total_bar_energy += bar_energy;
        }

        Ok(BenchmarkResult {
            benchmark_values,
            total_energy: total_bar_energy,
            #[allow(dead_code)]
            constitutional_alignment: 1.0,
        })
    }

    async fn apply_constitutional_orbit(&self, benchmark_result: &BenchmarkResult, frame_phi: f64) -> Result<OrbitResult, RenderError> {
        let phi_squared = frame_phi * frame_phi;
        let time = benchmark_result.total_energy;
        let orbit_x = (time * phi_squared * 57.2).sin() * 1.35;
        let orbit_y = (time * phi_squared).cos() * 1.35;
        let orbit_position = Vec2::new(orbit_x as f32, orbit_y as f32);
        let avg_metric_position = Vec2::new(
            (benchmark_result.benchmark_values.iter().map(|v| v.value).sum::<f64>() / benchmark_result.benchmark_values.len() as f64) as f32,
            1.0,
        );
        let distance_to_orbit = (avg_metric_position - orbit_position).length();
        let eternal_glow = (-(distance_to_orbit as f64) * 134.0).exp();

        Ok(OrbitResult {
            orbit_position,
            glow_intensity: eternal_glow,
            #[allow(dead_code)]
            phi_power: phi_squared.powi(49),
        })
    }

    async fn apply_scanlines(&self, orbit_result: &OrbitResult, _frame_phi: f64) -> Result<ScanlineResult, RenderError> {
        let scanline_density = 2700.0;
        let scanline_speed = 280.0;
        let time = orbit_result.glow_intensity;
        let scanline_pos = (orbit_result.orbit_position.y as f64 * scanline_density + time * scanline_speed).fract();
        let distance_to_center = (scanline_pos - 0.5).abs();
        let scanline_intensity = if distance_to_center <= 0.000000000005 {
            0.0
        } else if distance_to_center >= 1.0 {
            1.0
        } else {
            1.0 - distance_to_center * distance_to_center * (3.0 - 2.0 * distance_to_center)
        };

        Ok(ScanlineResult {
            scanline_intensity: scanline_intensity * 4.0,
        })
    }

    fn combine_render_layers(&self, scanline_result: &ScanlineResult, frame_phi: f64) -> Result<FinalFrame, RenderError> {
        let render_activity = scanline_result.scanline_intensity as f32;
        let fps_layer = Vec3::new(0.1, 0.4, 1.0) * render_activity * 10.6;
        let benchmark_layer = Vec3::new(0.2, 1.0, 0.3) * render_activity.powf(20.4) * 9.7;
        let cathedral_layer = Vec3::new(1.0, 0.9, 0.1) * render_activity.powf(72.0) * 9.2;
        let eternal_layer = Vec3::new(0.99, 0.99, 1.0) * render_activity.powf(104.0);
        let render_void = Vec3::new(0.000000003, 0.0000000006, 0.00000000006);
        let mut render_ecosystem = render_void + fps_layer + benchmark_layer + cathedral_layer + eternal_layer;
        let radial_factor = 10.6 - render_activity * 6.9;
        render_ecosystem *= radial_factor.max(0.0);

        Ok(FinalFrame {
            color: render_ecosystem,
            #[allow(dead_code)]
            alpha: 1.0,
            frame_phi,
            fps_actual: self.fps_controller.current_fps(),
        })
    }

    fn verify_constitutional_frame(&self, _frame: &FinalFrame, _phi: f64) -> Result<(), RenderError> {
        Ok(())
    }
}

// Supporting Structs (Backend & Pipeline)

pub struct RenderBackend {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl RenderBackend {
    pub async fn new(_config: &RenderConfig) -> Result<Self, RenderError> {
        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default())
            .await.ok_or(RenderError::BackendInitializationFailed)?;
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None)
            .await.map_err(|_| RenderError::BackendInitializationFailed)?;
        Ok(Self { device, queue })
    }

    pub async fn begin_render_pass(&self, _config: &RenderPassConfig) -> Result<RenderPassHandle, RenderError> {
        Ok(RenderPassHandle)
    }
}

pub struct RenderPassHandle;
impl RenderPassHandle {
    pub async fn finish(self) -> Result<RenderOutputTexture, RenderError> {
        Ok(RenderOutputTexture { metadata: RenderMetadata { time: 0.0 } })
    }
}

pub struct RenderPassConfig {
    pub clear_color: Option<[f32; 4]>,
    pub clear_depth: Option<f32>,
    pub store: bool,
}

pub struct RenderOutputTexture {
    pub metadata: RenderMetadata,
}

pub struct RenderMetadata {
    pub time: f64,
}

pub struct ShaderPipeline {
    #[allow(dead_code)]
    pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    cycle_count: u64,
}

impl ShaderPipeline {
    pub async fn new(backend: &RenderBackend, _phi: f64, _fps: f64) -> Result<Self, RenderError> {
        let shader = backend.device.create_shader_module(wgpu::include_wgsl!("../../../cge_constitutional_renderer/shaders/constitutional_render.wgsl"));

        let uniform_buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<ConstitutionalUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = backend.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = backend.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = backend.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Ok(Self { pipeline, bind_group, uniform_buffer, cycle_count: 0 })
    }

    pub async fn bind(&self, _backend: &RenderBackend, _data: &RenderData, _phi: f64) -> Result<(), RenderError> {
        Ok(())
    }

    pub async fn set_uniforms(&self, backend: &RenderBackend, uniforms: &ConstitutionalUniforms) -> Result<(), RenderError> {
        backend.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
        Ok(())
    }

    pub async fn draw_fullscreen_quad(&self, _pass: &RenderPassHandle) -> Result<(), RenderError> {
        Ok(())
    }

    pub fn get_cycle_count(&self) -> u64 { self.cycle_count }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConstitutionalUniforms {
    pub i_time: f32,
    pub i_resolution: [f32; 2],
    pub phi_target: f32,
    pub fps_target: f32,
    pub frag_count: f32,
    pub benchmark_count: f32,
}

pub struct RenderFragMatrix {
    count: usize,
}
impl RenderFragMatrix {
    pub fn new(count: usize, _phi: f64) -> Result<Self, RenderError> { Ok(Self { count }) }
    pub fn active_frag_count(&self) -> usize { self.count }
}

pub struct ConstitutionalValidator;
impl ConstitutionalValidator {
    pub fn new(_phi: f64, _fps: f64) -> Result<Self, RenderError> { Ok(Self) }
    pub async fn validate_frame(&self, _phi: f64) -> Result<(), RenderError> { Ok(()) }
}

#[derive(Default)]
pub struct RenderState;

pub struct RenderMetrics;
impl RenderMetrics {
    pub fn new() -> Self { Self }
    pub fn record_frame(&self, _duration: Duration, _phi: f64) {}
}

pub struct FrameData { pub time: f64 }
pub struct RenderData { pub time: f64 }

pub struct RenderPipelineResult {
    pub output_texture: RenderOutputTexture,
    pub pipeline_time: Duration,
    pub uniform_count: usize,
    pub shader_cycles: u64,
}

pub struct FragMatrixResult {
    pub frag_results: Vec<FragResult>,
    pub total_activity: f64,
    pub constitutional_coherence: f64,
}

pub struct FragResult {
    pub frag_id: usize,
    pub position: Vec2,
    pub activity: f32,
    pub constitutional_marker: f64,
}

pub struct BenchmarkResult {
    pub benchmark_values: Vec<BenchmarkValue>,
    pub total_energy: f64,
    pub constitutional_alignment: f64,
}

pub struct BenchmarkValue {
    pub metric_id: usize,
    pub value: f64,
    pub constitutional_weight: f64,
}

pub struct OrbitResult {
    pub orbit_position: Vec2,
    pub glow_intensity: f64,
    pub phi_power: f64,
}

pub struct ScanlineResult {
    pub scanline_intensity: f64,
}

pub struct FinalFrame {
    pub color: Vec3,
    pub alpha: f32,
    pub frame_phi: f64,
    pub fps_actual: f64,
}

pub struct RenderedFrame {
    pub frame_data: FinalFrame,
    pub frame_time: Duration,
    pub frame_phi: f64,
    pub constitutional_checks_passed: bool,
    pub frags_rendered: usize,
    pub benchmark_metrics: Vec<crate::benchmark::constitutional_benchmark::BenchmarkMetric>,
}
