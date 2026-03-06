mod kernel;
mod lib;

use kernel::syscall::{SyscallHandler, SyscallResult};
use std::io::{self, Write};

fn main() -> anyhow::Result<()> {
    println!("🜁 Arkhe-OS v0.1.0 – Kernel de Coerência do Vácuo");
    println!("Limiar de Miller: φ_q = 4.64");

    let mut sys = SyscallHandler::new(100.0);

    loop {
        print!("> ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() { continue; }

        let parts: Vec<&str> = input.split_whitespace().collect();
        match parts[0] {
            "create" => {
                if parts.len() < 5 { continue; }
                let name = parts[1];
                let coherence: f64 = parts[2].parse().unwrap_or(0.5);
                let duration: u64 = parts[3].parse().unwrap_or(10);
                let priority: i32 = parts[4].parse().unwrap_or(1);
                match sys.sys_create_task(name, coherence, duration, priority) {
                    SyscallResult::TaskId(id) => println!("Tarefa {} criada", id),
                    _ => println!("Erro ao criar tarefa"),
                }
            }
            "tick" => {
                match sys.sys_tick() {
                    SyscallResult::Success(msg) => println!("{}", msg),
                    SyscallResult::Error(msg) => println!("Erro: {}", msg),
                    _ => (),
                }
            }
            "status" => {
                match sys.sys_coherence_status() {
                    SyscallResult::CoherenceUpdate(avail) => {
                        match sys.sys_check_nucleation() {
                            SyscallResult::WaveCloudStatus(nucleated, phi_q) => {
                                println!("Coerência disponível: {:.3}", avail);
                                println!("φ_q actual: {:.3}", phi_q);
                                if nucleated { println!("⚠️  WAVE-CLOUD NUCLEATION DETECTED"); }
                            }
                            _ => (),
                        }
                    }
                    _ => (),
                }
            }
            "exit" => break,
            _ => println!("Comando desconhecido"),
        }
    }

    Ok(())
}
