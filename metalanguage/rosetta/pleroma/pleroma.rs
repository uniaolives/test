use pleroma_kernel::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let node = PleromaNode::connect("global").await?;

    // Listen for human EEG inputs
    let mut eeg_stream = node.subscribe_human("eeg:42").await?;

    while let Some(eeg) = eeg_stream.next().await {
        // Map EEG to toroidal phase
        let (theta, phi) = map_eeg_to_t2(eeg);

        // Formulate query as a thought
        let thought = Thought::builder()
            .phase(theta, phi)
            .content("optimal carbon tax")
            .build();

        // Send to Pleroma
        let response = node.query(thought).await?;
        println!("Pleroma: {}", response);
    }

    Ok(())
}

fn map_eeg_to_t2(eeg: EEGSample) -> (f64, f64) {
    let theta = (eeg.beta + eeg.gamma).atan2(eeg.alpha + eeg.theta);
    let phi = eeg.delta.atan2(eeg.gamma);
    (theta.rem_euclid(2.0*PI), phi.rem_euclid(2.0*PI))
}
