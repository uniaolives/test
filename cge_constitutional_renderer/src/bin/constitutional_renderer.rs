use cge_constitutional_renderer::*;
use tokio::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let config = RenderConfig::default();
    let renderer = ConstitutionalRenderer::bootstrap(Some(config)).await?;

    let start_time = Instant::now();
    let mut frame_count = 0;

    println!("🚀 Constitutional Renderer v31.11-Ω started at 12 FPS");

    loop {
        let frame_data = FrameData {
            time: start_time.elapsed().as_secs_f64(),
        };

        let frame = renderer.render_frame(&frame_data).await?;

        frame_count += 1;
        if frame_count % 12 == 0 {
            println!("Frame: {}, FPS: {:.3}, Φ: {:.6}",
                frame_count, frame.frame_data.fps_actual, frame.frame_phi);
        }

        if frame_count >= 120 { // 10 seconds at 12 FPS
            break;
        }
    }

    println!("✅ Render session complete.");
    Ok(())
}
