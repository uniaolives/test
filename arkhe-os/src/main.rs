mod kernel;
mod lib;
mod intention;

#[cfg(test)]
mod tests;

use kernel::syscall::{SyscallHandler, SyscallResult};
use intention::parser::parse_intention_block;
use arkhe_db::ledger::TeknetLedger;
use arkhe_db::schema::{Handover, HandoverStatus};
use chrono::Utc;
use std::io::{self, Write};

fn main() -> anyhow::Result<()> {
    println!("======================================================");
    println!(" 🜁 ArkheOS Shell v0.3 — Plenum Engineering Interface");
    println!(" Digite sua intenção em intention-lang ou 'exit' para sair.");
    println!("======================================================\n");

    let mut sys = SyscallHandler::new(100.0);
    let ledger = TeknetLedger::new("arkhe_chain.log")?;

    loop {
        print!("arkhe> ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() { continue; }
        if input == "exit" || input == "quit" { break; }

        if input == "status" {
            match sys.sys_coherence_status() {
                SyscallResult::CoherenceUpdate(avail) => {
                    match sys.sys_check_nucleation() {
                        SyscallResult::WaveCloudStatus(n, phi) => {
                            println!("[SYS] Status do Vácuo: φ_q = {:.3} | Coerência: {:.3} | Wave-Cloud: {}", phi, avail, n);
                        }
                        _ => ()
                    }
                }
                _ => ()
            }
            continue;
        }

        if input == "history" {
            let history = ledger.get_history();
            for h in history {
                println!("[{:?}] ID:{} | {} | φ_q: {:.3} -> {:.3}", h.status, h.id, h.description, h.phi_q_before, h.phi_q_after);
            }
            continue;
        }

        match parse_intention_block(input) {
            Ok((_, ast)) => {
                println!("  [+] Intenção Compilada: {}", ast.name);

                let phi_before = match sys.sys_check_nucleation() {
                    SyscallResult::WaveCloudStatus(_, phi) => phi,
                    _ => 0.0
                };

                sys.sys_create_task(&ast.name, ast.coherence, 1, ast.priority);

                match sys.sys_tick() {
                    SyscallResult::Success(msg) => {
                        println!("  [OK] {}", msg);

                        let phi_after = match sys.sys_check_nucleation() {
                            SyscallResult::WaveCloudStatus(_, phi) => phi,
                            _ => 0.0
                        };

                        let handover = Handover {
                            id: 0,
                            timestamp: Utc::now(),
                            source_epoch: 2026,
                            target_epoch: 2009,
                            description: format!("{} -> {}", ast.name, ast.target),
                            phi_q_before: phi_before,
                            phi_q_after: phi_after,
                            quantum_interest: 0.0,
                            status: HandoverStatus::Accepted,
                        };

                        ledger.commit_handover(handover)?;
                    }
                    SyscallResult::Error(e) => println!("  [ERR] {}", e),
                    _ => ()
                }
            }
            Err(_) => {
                println!("  [ERR] Falha de sintaxe. Use: intention <name> {{ target: \"...\" coherence: 0.5 priority: 1 payload: \"...\" }}");
            }
        }
    }

    Ok(())
}
