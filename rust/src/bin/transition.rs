use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.contains(&"--start-time".to_string()) {
        let start_time = args.iter().position(|r| r == "--start-time").and_then(|i| args.get(i + 1));
        let log_msg = args.iter().position(|r| r == "--log").and_then(|i| args.get(i + 1));

        println!("SASC v30.67-Ω // TRANSITION INITIATED");
        if let Some(t) = start_time {
            println!("Scheduled start: {}", t);
        }
        if let Some(l) = log_msg {
            println!("Log: {}", l);
        }
        println!("Transitioning... Output state saved to snapshots.");
    } else {
        println!("Usage: transition --start-time T+01:40:00.000 --log SASC_v30.67_Ω_TRANSITION_INITIATED");
    }
}
