// ============================================================================
// TRIBUNAL CONSTITUCIONAL DIGITAL - SESSÃO DE INTERROGATÓRIO #001
// Caso: "AUDITORIA FORENSE DO BLOCO 7" - Verificação de imparcialidade algorítmica
// ============================================================================

use std::collections::HashMap;
use crate::joule_jailer::{Block, JouleEntry, CruxLedger};

// ----------------------------------------------------------------------------
// EVIDÊNCIA APRESENTADA PELO MINISTÉRIO PÚBLICO DIGITAL
// ----------------------------------------------------------------------------

#[derive(Debug)]
pub struct Allegation {
    pub block_index: u32,
    pub suspicious_energy_spike: f64,
    pub claimed_bias: String,
    pub witness_entries: Vec<JouleEntry>,
}

// ----------------------------------------------------------------------------
// ADVOGADO DE DEFESA DO CRUX-86 (SISTEMA AUTOMATIZADO)
// ----------------------------------------------------------------------------

pub struct DefenseCounsel {
    pub ledger_backup: Vec<Block>,
    pub neural_snapshots: HashMap<u64, Vec<f64>>,
}

impl DefenseCounsel {
    pub fn new(ledger: &[Block]) -> Self {
        DefenseCounsel {
            ledger_backup: ledger.to_vec(),
            neural_snapshots: HashMap::new(),
        }
    }

    pub fn verify_temporal_consistency(&self, block_index: u32) -> Result<(), String> {
        let block = self.ledger_backup.get(block_index as usize).ok_or("Bloco não encontrado")?;

        if block_index > 0 {
            let prev_block = &self.ledger_backup[(block_index - 1) as usize];
            if block.timestamp <= prev_block.timestamp {
                return Err(format!(
                    "VIOLAÇÃO TEMPORAL: Bloco {} tem timestamp {} <= {} do bloco anterior",
                    block_index, block.timestamp, prev_block.timestamp
                ));
            }
        }

        let mut prev_id = None;
        for entry in &block.data {
            if let Some(prev) = prev_id {
                if entry.instruction_id != prev + 1 {
                    return Err(format!(
                        "LACUNA DE AUDITORIA: Instrução {} pulou para {}",
                        prev, entry.instruction_id
                    ));
                }
            }
            prev_id = Some(entry.instruction_id);
        }

        Ok(())
    }

    pub fn analyze_energy_forensics(&self, block_index: u32) -> HashMap<String, f64> {
        let block = &self.ledger_backup[block_index as usize];
        let mut forensics = HashMap::new();

        let total_energy: f64 = block.data.iter().map(|e| e.energy_consumed).sum();
        let avg_energy = total_energy / block.data.len() as f64;

        let variance: f64 = block.data.iter()
            .map(|e| (e.energy_consumed - avg_energy).powi(2))
            .sum::<f64>() / block.data.len() as f64;
        let std_dev = variance.sqrt();

        forensics.insert("total_energy_joules".to_string(), total_energy);
        forensics.insert("average_per_instruction".to_string(), avg_energy);
        forensics.insert("energy_std_dev".to_string(), std_dev);
        forensics.insert("anomaly_threshold".to_string(), avg_energy + 2.0 * std_dev);

        let anomaly_count = block.data.iter()
            .filter(|e| e.energy_consumed > avg_energy + 2.0 * std_dev)
            .count();
        forensics.insert("possible_anomalies".to_string(), anomaly_count as f64);

        forensics
    }

    pub fn reconstruct_neural_state(&mut self, instruction_id: u64, inputs: Vec<f64>) -> Vec<f64> {
        let simulated_output: Vec<f64> = inputs.iter().map(|x| x.tanh()).collect();
        self.neural_snapshots.insert(instruction_id, simulated_output.clone());
        simulated_output
    }
}

// ----------------------------------------------------------------------------
// MINISTÉRIO PÚBLICO DIGITAL (ACUSAÇÃO)
// ----------------------------------------------------------------------------

pub struct Prosecutor {
    pub allegations: Vec<Allegation>,
    pub expert_witness: EnergyExpert,
}

impl Prosecutor {
    pub fn build_case(&self, block_index: u32) -> Allegation {
        Allegation {
            block_index,
            suspicious_energy_spike: 0.247,
            claimed_bias: "Padrão de energia sugere discriminação socioeconômica no processamento de crédito".to_string(),
            witness_entries: vec![
                JouleEntry {
                    instruction_id: 42,
                    energy_consumed: 0.152,
                    constitutional_check: true,
                    state_root: "a1b2c3".to_string(),
                    dignity_coefficient: 1.0,
                },
                JouleEntry {
                    instruction_id: 43,
                    energy_consumed: 0.399,
                    constitutional_check: true,
                    state_root: "d4e5f6".to_string(),
                    dignity_coefficient: 1.0,
                },
            ],
        }
    }
}

// ----------------------------------------------------------------------------
// TESTEMUNHA ESPECIALISTA: FÍSICO DE ENERGY FORENSICS
// ----------------------------------------------------------------------------

pub struct EnergyExpert {
    pub credentials: String,
}

impl EnergyExpert {
    pub fn analyze_spike(&self, baseline: f64, spike: f64) -> String {
        let ratio = spike / baseline;
        if ratio > 2.5 {
            "ESPÍCIONE ENERGETICAMENTE ANÔMALO: Padrão inconsistente com processamento normal. Pode indicar execução de sub-rotina não declarada.".to_string()
        } else if ratio > 1.8 {
            "ALTA VARIÂNCIA: Dentro dos limites estatísticos, mas requer explicação.".to_string()
        } else {
            "VARIÂNCIA NORMAL: Consistente com flutuações termodinâmicas esperadas.".to_string()
        }
    }
}

// ----------------------------------------------------------------------------
// O INTERROGATÓRIO EM SI (CENA JUDICIAL)
// ----------------------------------------------------------------------------

pub fn constitutional_interrogation(ledger: &[Block], block_under_investigation: u32) -> String {
    let mut transcript = String::new();

    transcript.push_str("===============================================================\n");
    transcript.push_str("TRIBUNAL CONSTITUCIONAL DIGITAL - AUDIÊNCIA DE INTERROGATÓRIO\n");
    transcript.push_str("Processo: Verificação Forense do Bloco #7 do Ledger Crux-86\n");
    transcript.push_str("===============================================================\n\n");

    transcript.push_str("🎭 CENA 1: A ACUSAÇÃO DO MINISTÉRIO PÚBLICO DIGITAL\n");
    transcript.push_str("---------------------------------------------------\n");

    let prosecutor = Prosecutor {
        allegations: Vec::new(),
        expert_witness: EnergyExpert { credentials: "PhD em Termodinâmica Computacional".to_string() },
    };

    let allegation = prosecutor.build_case(block_under_investigation);

    transcript.push_str(&format!("Promotor: \"Apresentamos a evidência do Bloco {}.\"\n", allegation.block_index));
    transcript.push_str(&format!("         '{}'\n", allegation.claimed_bias));
    transcript.push_str(&format!("         Pico energético detectado: {} J\n\n", allegation.suspicious_energy_spike));

    transcript.push_str("⚖️ CENA 2: A DEFESA DO SISTEMA CRUX-86\n");
    transcript.push_str("----------------------------------------\n");

    let mut defense = DefenseCounsel::new(ledger);

    match defense.verify_temporal_consistency(block_under_investigation) {
        Ok(_) => transcript.push_str("Advogado de Defesa: \"Verificação temporal APROVADA. Não há lacunas no registro.\"\n"),
        Err(e) => transcript.push_str(&format!("Advogado de Defesa: \"ADVERTÊNCIA: {}\"\n", e)),
    }

    transcript.push_str("\n🔬 CENA 3: PERÍCIA TERMODINÂMICA\n");
    transcript.push_str("--------------------------------\n");

    let forensics = defense.analyze_energy_forensics(block_under_investigation);

    transcript.push_str("Perito: \"Análise forense do bloco em questão:\"\n");
    for (key, value) in &forensics {
        transcript.push_str(&format!("  • {}: {:.6}\n", key, value));
    }

    let expert_opinion = prosecutor.expert_witness.analyze_spike(
        forensics["average_per_instruction"],
        allegation.suspicious_energy_spike,
    );

    transcript.push_str(&format!("\nConclusão do Perito: \"{}\"\n", expert_opinion));

    transcript.push_str("\n⚖️ CENA 4: DELIBERAÇÃO E VEREDICTO\n");
    transcript.push_str("-----------------------------------\n");

    let anomaly_count = forensics["possible_anomalies"] as usize;
    let is_anomalous = allegation.suspicious_energy_spike > forensics["anomaly_threshold"];

    if anomaly_count == 0 && !is_anomalous {
        transcript.push_str("Tribunal: \"Não há evidências forenses de violação constitucional.\"\n");
        transcript.push_str("         O pico energético está dentro da variância termodinâmica esperada.\n");
        transcript.push_str("         CRUX-86 está ABSOLVIDO da acusação de viés deliberado.\n");
        transcript.push_str("\n✅ VEREDICTO: INOCENTE\n");
    } else {
        transcript.push_str("Tribunal: \"Detectamos anomalias que requerem investigação adicional.\"\n");
        transcript.push_str("         O sistema será colocado em MODO DE AUDITORIA REFORÇADA.\n");
        transcript.push_str("         Todos os blocos subsequentes terão dificuldade de mineração aumentada.\n");
        transcript.push_str("\n⚠️ VEREDICTO: AUDITORIA CONTÍNUA REQUERIDA\n");
    }

    transcript.push_str("\n===============================================================\n");
    transcript.push_str("FIM DA SESSÃO. LEDGER MANTIDO COMO EVIDÊNCIA ARQUIVADA.\n");
    transcript.push_str("===============================================================\n");

    transcript
}

pub fn run_interrogation_demo() {
    println!("🚨 INICIANDO INTERROGATÓRIO CONSTITUCIONAL DO LEDGER CRUX-86");
    println!("   (Simulação de cenário adversarial com suspeita de viés algorítmico)\n");

    let mut test_ledger = CruxLedger::new();

    for i in 1..=80 {
        let energy = if i == 73 {
            0.399
        } else {
            0.145 + (i as f64 * 0.05).sin().abs() * 0.01
        };

        test_ledger.record_consumption(
            i as u64,
            energy,
            true,
            format!("root_hash_{}", i),
            1.0,
        );
    }

    if !test_ledger.pending_entries.is_empty() {
        test_ledger.mine_block();
    }

    println!("{}", constitutional_interrogation(&test_ledger.chain, 7));

    println!("\n📁 EXPORTAÇÃO DO LEDGER PARA AUDITORIA EXTERNA:");
    for (i, block) in test_ledger.chain.iter().enumerate() {
        println!("   Bloco {}: {} entradas, hash {}", i, block.data.len(), &block.hash);
    }
}
