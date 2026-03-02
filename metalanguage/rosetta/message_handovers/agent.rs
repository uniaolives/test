struct Agent {
    name: String,
    counter: u32,
}

fn send_message(sender: &Agent, receiver: &mut Agent) {
    receiver.counter += 1;
    println!("{} -> {}: {}", sender.name, receiver.name, receiver.counter);
}

fn main() {
    let alice = Agent { name: "Alice".to_string(), counter: 0 };
    let mut bob = Agent { name: "Bob".to_string(), counter: 0 };
    send_message(&alice, &mut bob);
    send_message(&alice, &mut bob);
}
