use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use parking_lot::Mutex;

#[derive(Debug, thiserror::Error)]
pub enum TimingError {
    #[error("FPS inválido: {0}")]
    InvalidFps(f64),
    #[error("Φ inválido: {0}")]
    InvalidPhi(f64),
    #[error("Desvio de Φ: {0}, esperado {1}")]
    PhiDeviation(f64, f64),
    #[error("Violação de timing")]
    TimingViolation,
}

pub struct ConstitutionalFpsController {
    target_fps: f64,
    target_frame_time: Duration,
    phi_target: f64,
    start_time: Instant,
    frame_counter: AtomicU64,
    last_frame_time: Mutex<Option<Instant>>,
    frame_times: Mutex<Vec<Duration>>,
    average_fps: Mutex<f64>,
    #[allow(dead_code)]
    min_frame_time: AtomicU64,
    #[allow(dead_code)]
    max_frame_time: AtomicU64,
}

impl ConstitutionalFpsController {
    pub fn new(target_fps: f64, phi_target: f64) -> Result<Self, TimingError> {
        if (target_fps - 12.0).abs() > 0.001 {
            return Err(TimingError::InvalidFps(target_fps));
        }
        if (phi_target - phi_calculus::PHI_TARGET).abs() > 0.001 {
            return Err(TimingError::InvalidPhi(phi_target));
        }
        let target_frame_time = Duration::from_secs_f64(1.0 / target_fps);
        Ok(Self {
            target_fps,
            target_frame_time,
            phi_target,
            start_time: Instant::now(),
            frame_counter: AtomicU64::new(0),
            last_frame_time: Mutex::new(None),
            frame_times: Mutex::new(Vec::with_capacity(1000)),
            average_fps: Mutex::new(0.0),
            min_frame_time: AtomicU64::new(u64::MAX),
            max_frame_time: AtomicU64::new(0),
        })
    }

    pub async fn wait_for_frame_timing(&self) -> Result<(), TimingError> {
        let frame_start = Instant::now();
        let _frame_number = self.frame_counter.fetch_add(1, Ordering::SeqCst);
        let frame_phi = self.calculate_frame_phi(frame_start)?;
        let wait_time = self.calculate_wait_time(frame_start, frame_phi).await?;
        if !wait_time.is_zero() {
            sleep(wait_time).await;
        }
        let frame_end = Instant::now();
        let frame_duration = frame_end.duration_since(frame_start);
        self.update_metrics(frame_duration);
        Ok(())
    }

    fn calculate_frame_phi(&self, frame_time: Instant) -> Result<f64, TimingError> {
        let elapsed = frame_time.duration_since(self.start_time).as_secs_f64();
        Ok(phi_calculus::get_frame_phi(elapsed, self.target_fps))
    }

    async fn calculate_wait_time(&self, frame_start: Instant, frame_phi: f64) -> Result<Duration, TimingError> {
        let mut last_frame = self.last_frame_time.lock();
        if let Some(last_time) = *last_frame {
            let time_since_last = frame_start.duration_since(last_time);
            if time_since_last >= self.target_frame_time {
                *last_frame = Some(frame_start);
                return Ok(Duration::ZERO);
            }
            let remaining = self.target_frame_time - time_since_last;
            let phi_adjustment = 1.0 + (frame_phi - self.phi_target) * 0.1;
            let adjusted_wait = remaining.mul_f64(phi_adjustment);
            *last_frame = Some(frame_start + adjusted_wait);
            Ok(adjusted_wait)
        } else {
            *last_frame = Some(frame_start);
            Ok(Duration::ZERO)
        }
    }

    fn update_metrics(&self, duration: Duration) {
        let mut times = self.frame_times.lock();
        times.push(duration);
        if times.len() > 1000 { times.remove(0); }
        let avg = times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / times.len() as f64;
        *self.average_fps.lock() = 1.0 / avg;
    }

    pub fn current_fps(&self) -> f64 {
        *self.average_fps.lock()
    }
}
