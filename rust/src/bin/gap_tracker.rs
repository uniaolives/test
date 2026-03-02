use sasc_core::safety::phase_2_enhanced::Phase2EnhancedSafety;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.contains(&"--monitor".to_string()) {
        let safety = Phase2EnhancedSafety::new();
        println!("Deploying Gap Tracker for target: {}", safety.gap_tracker.target);
        println!("Update frequency: {:?}", safety.gap_tracker.update_frequency);
        println!("Field team coordination: {}", safety.gap_tracker.field_team_coordination);
        println!("Nexus suggestion only mode active.");
    } else {
        println!("Usage: gap_tracker --monitor madeira_unmonitored_tributary ...");
    }
}
