//! Interface de linha de comando para o arkhe-os.
//! Permite criar tarefas e executar ciclos de escalonamento.

mod kernel;
mod physics;

use kernel::syscall::SyscallHandler;
use std::io::{self, Write};

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("🜁 Arkhe-OS v0.1.0 – Kernel de Coerência do Vácuo");
    println!("Limiar de Miller: φ_q = 4.64");
    println!("Comandos: create <nome> <coerência> <duração> <prioridade> | tick | status | nucleation | handover <época> <payload> | exit");

    let mut sys = SyscallHandler::new(100.0); // Coerência inicial de 100 unidades
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

        if input.is_empty() {
            continue;
        }
        if input.is_empty() { continue; }

        let parts: Vec<&str> = input.split_whitespace().collect();
        match parts[0] {
            "create" => {
                if parts.len() < 5 {
                    println!("Uso: create <nome> <coerência> <duração> <prioridade>");
                    continue;
                }
                if parts.len() < 5 { continue; }
                let name = parts[1];
                let coherence: f64 = parts[2].parse().unwrap_or(0.5);
                let duration: u64 = parts[3].parse().unwrap_or(10);
                let priority: i32 = parts[4].parse().unwrap_or(1);
                match sys.sys_create_task(name, coherence, duration, priority) {
                    kernel::syscall::SyscallResult::TaskId(id) => {
                        println!("Tarefa criada com ID {}", id);
                    }
                    SyscallResult::TaskId(id) => println!("Tarefa {} criada", id),
                    _ => println!("Erro ao criar tarefa"),
                }
            }
            "tick" => {
                match sys.sys_tick() {
                    kernel::syscall::SyscallResult::Success(msg) => println!("{}", msg),
                    kernel::syscall::SyscallResult::Error(msg) => println!("Erro: {}", msg),
                    SyscallResult::Success(msg) => println!("{}", msg),
                    SyscallResult::Error(msg) => println!("Erro: {}", msg),
                    _ => (),
                }
            }
            "status" => {
                match sys.sys_coherence_status() {
                    kernel::syscall::SyscallResult::CoherenceUpdate(avail) => {
                        // Para obter φ_q, usamos sys_check_nucleation
                        let nucl = sys.sys_check_nucleation();
                        let (nucleated, phi_q) = match nucl {
                            kernel::syscall::SyscallResult::WaveCloudStatus(n, p) => (n, p),
                            _ => (false, 0.0),
                        };
                        println!("Coerência disponível: {:.3}", avail);
                        println!("φ_q actual: {:.3}", phi_q);
                        if nucleated {
                            println!("⚠️  WAVE-CLOUD NUCLEATION DETECTED");
                        }
                    }
                    _ => (),
                }
            }
            "nucleation" => {
                match sys.sys_check_nucleation() {
                    kernel::syscall::SyscallResult::WaveCloudStatus(nucleated, phi_q) => {
                        println!("φ_q = {:.3} – {}", phi_q, if nucleated { "NUCLEADO" } else { "abaixo do limiar" });
                    }
                    _ => (),
                }
            }
            "handover" => {
                if parts.len() < 3 {
                    println!("Uso: handover <época> <payload>");
                    continue;
                }
                let epoch: u32 = parts[1].parse().unwrap_or(2009);
                let payload = parts[2..].join(" ");
                match sys.sys_handover(epoch, &payload) {
                    kernel::syscall::SyscallResult::Success(msg) => println!("{}", msg),
                    _ => println!("Erro no handover"),
                }
            }
            "exit" | "quit" => break,
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
