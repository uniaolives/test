use bitcoin::block::{Header as BlockHeader, BlockHash};
use bitcoin::hashes::Hash;
use bitcoin::Network;
use nalgebra::DVector;
use crate::oloid::OloidState;
use log::info;

pub mod sanity;

/// O Cristal de Tempo como Fonte de Verdade
pub struct TimechainAnchor {
    /// O hash do último bloco conhecido (O "Agora" absoluto)
    pub last_tip: BlockHash,
    /// A "massa" temporal acumulada (Difficulty approximation)
    pub accumulated_work: u128,
    /// Oloid acoplado
    oloid_link: OloidState,
}

impl TimechainAnchor {
    pub fn new(oloid: OloidState) -> Self {
        Self {
            last_tip: BlockHash::all_zeros(), // Gênesis placeholder
            accumulated_work: 0,
            oloid_link: oloid,
        }
    }

    /// Ingere um novo bloco da Timechain (Um "tique" do relógio universal)
    /// Isso atua como um 'Handover Externo' forçado.
    pub fn process_new_block(&mut self, header: BlockHeader) {
        // 1. Validação Temporal (Prova de Trabalho)
        // Usamos a dificuldade como medida de "densidade de realidade".
        // Network enum is usually what's expected for network parameters.
        let diff = header.difficulty(Network::Bitcoin);
        self.accumulated_work += diff;
        self.last_tip = header.block_hash();

        // 2. Extração de Entropia Cristalina
        let entropy_source = header.merkle_root.to_byte_array();

        // 3. Sincronização do Oloid
        self.entangle_with_timechain(&entropy_source);
    }

    fn entangle_with_timechain(&mut self, entropy: &[u8; 32]) {
        // Converte o hash em um vetor de estado
        let injection_vector = DVector::from_iterator(
            32,
            entropy.iter().map(|&b| b as f64 / 255.0)
        );

        // Perturbação controlada no Oloid Core
        self.oloid_link.inject_external_rhythm(injection_vector);

        info!(
            "⏳ TIMECHAIN SYNC: Bloco processado. Accumulated Difficulty: {}. Oloid realinhado.",
            self.accumulated_work
        );
    }
}
