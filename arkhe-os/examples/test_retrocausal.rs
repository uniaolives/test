use arkhe_os::propagation::payload::OrbPayload;

fn main() {
    let now = 1710115200; // Mock current time
    let future = now + 3600;
    let past = now - 3600;

    let retro_orb = OrbPayload::create(1.0, 1.0, 0.1, now, future, None, None);
    let causal_orb = OrbPayload::create(1.0, 1.0, 0.1, now, past, None, None);

    println!("Testing Retrocausal Targeting Mode:");
    println!("Orb targeting FUTURE (+1h): is_retrocausal = {}", retro_orb.is_retrocausal());
    println!("Orb targeting PAST (-1h): is_retrocausal = {}", causal_orb.is_retrocausal());

    if retro_orb.is_retrocausal() && !causal_orb.is_retrocausal() {
        println!("✅ Coherence logic verified: Future targets are identified as retrocausal attractors.");
    } else {
        println!("❌ Coherence logic mismatch.");
    }
}
