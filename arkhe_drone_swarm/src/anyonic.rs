//! Módulo Anyonic – Estatística Fracionária e Braiding Topológico
//!
//! Implementa:
//! - Parâmetro anyónico α ∈ [0,1] (fração exata)
//! - Fase complexa acumulada por handovers
//! - Operações de braiding em 1D (troca entre handovers adjacentes)
//! - Cálculo de dissipação D_n(H) ~ k⁻ⁿ⁻¹
//! - Detecção de vórtices anyónicos (nós com fase não trivial)

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::f64::consts::PI;

/// ID único para um handover
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HandoverId(pub u64);

/// Estatística anyónica representada como fração exata entre 0 e 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AnyonStatistic {
    numerator: u64,
    denominator: u64,
}

impl AnyonStatistic {
    /// Cria uma nova estatística a partir de uma fração.
    /// Ex: AnyonStatistic::new(1, 3) para α = 1/3.
    pub fn new(numerator: u64, denominator: u64) -> Result<Self, &'static str> {
        if denominator == 0 {
            return Err("Denominator cannot be zero");
        }
        if numerator > denominator {
            return Err("α must be in [0,1]");
        }
        Ok(Self {
            numerator,
            denominator,
        })
    }

    /// Cria a partir de um float (aproximado). Útil para testes.
    pub fn from_f64(alpha: f64) -> Result<Self, &'static str> {
        if !(0.0..=1.0).contains(&alpha) {
            return Err("α must be in [0,1]");
        }
        // Simplificação simples para denominador até 1000
        let mut best_num = 0;
        let mut best_den = 1;
        let mut best_err = (alpha - 0.0).abs();
        for den in 1..=1000 {
            let num = (alpha * den as f64).round() as u64;
            if num > den {
                continue;
            }
            let err = (alpha - (num as f64 / den as f64)).abs();
            if err < best_err {
                best_num = num;
                best_den = den;
                best_err = err;
            }
        }
        Ok(Self {
            numerator: best_num,
            denominator: best_den,
        })
    }

    /// Valor como f64.
    pub fn as_f64(&self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }

    /// Fase de braiding para uma troca (e^(iπ·α)).
    pub fn braiding_phase(&self) -> Complex64 {
        Complex64::from_polar(1.0, PI * self.as_f64())
    }

    /// Estatística de troca entre dois ányons (média das fases).
    pub fn exchange_phase(&self, other: &AnyonStatistic) -> Complex64 {
        let avg = (self.as_f64() + other.as_f64()) / 2.0;
        Complex64::from_polar(1.0, PI * avg)
    }

    /// Retorna true se α = 0 (bosónico).
    pub fn is_bosonic(&self) -> bool {
        self.numerator == 0
    }

    /// Retorna true se α = 1 (fermiónico).
    pub fn is_fermionic(&self) -> bool {
        self.numerator == self.denominator
    }

    /// Retorna true se 0 < α < 1 (anyónico).
    pub fn is_anyonic(&self) -> bool {
        self.numerator > 0 && self.numerator < self.denominator
    }
}

/// Handover topológico com memória de fase anyónica.
#[derive(Debug, Clone)]
pub struct TopologicalHandover {
    pub id: HandoverId,
    pub node_i: String,
    pub node_j: String,
    pub alpha: AnyonStatistic,          // estatística do handover (média dos nós)
    pub accumulated_phase: Complex64,    // fase total adquirida por braiding
    pub braid_partners: Vec<HandoverId>, // histórico de handovers com quem trocou
    pub timestamp: u64,
    pub intensity: f64,
}

impl TopologicalHandover {
    pub fn new(
        id: HandoverId,
        node_i: String,
        node_j: String,
        alpha: AnyonStatistic,
        timestamp: u64,
        intensity: f64,
    ) -> Self {
        Self {
            id,
            node_i,
            node_j,
            alpha,
            accumulated_phase: Complex64::new(1.0, 0.0),
            braid_partners: Vec::new(),
            timestamp,
            intensity,
        }
    }

    /// Troca anyónica com outro handover (braiding).
    /// Retorna a fase adquirida por este handover.
    pub fn braid_with(&mut self, other: &mut TopologicalHandover) -> Complex64 {
        // Verificar se compartilham um nó (adjacência 1D)
        let share_i = self.node_i == other.node_i || self.node_i == other.node_j;
        let share_j = self.node_j == other.node_i || self.node_j == other.node_j;
        if !share_i && !share_j {
            // Handovers disjuntos: comutam, fase 1
            return Complex64::new(1.0, 0.0);
        }

        // Calcular fase de troca
        let phase = self.alpha.exchange_phase(&other.alpha);

        // Atualizar fases acumuladas
        self.accumulated_phase *= phase;
        other.accumulated_phase *= phase.conj(); // conjugada para conservar produto total

        // Registrar parceiro
        self.braid_partners.push(other.id);
        other.braid_partners.push(self.id);

        phase
    }

    /// Calcula a dissipação para um dado momento k e número de corpos n.
    /// D_n ∼ C(α) * k^(-n-1) * |F̃(k)|², onde |F̃(k)|² é assumido 1 por simplicidade.
    pub fn dissipation(&self, k: f64, n_body: usize) -> f64 {
        let universal = k.powi(-(n_body as i32 + 1));
        if n_body == 2 {
            // Universal, independente de α
            universal
        } else {
            // Coeficiente depende da fase acumulada
            let coeff = self.accumulated_phase.norm().powi(n_body as i32 - 2);
            coeff * universal
        }
    }
}

/// Hipergrafo anyónico, gerenciando nós e handovers.
pub struct AnyonicHypergraph {
    /// Estatística de cada nó.
    node_stat: BTreeMap<String, AnyonStatistic>,
    /// Handovers ativos, indexados por ID.
    handovers: BTreeMap<HandoverId, TopologicalHandover>,
    /// Próximo ID disponível.
    next_id: u64,
    /// Invariante topológico global (winding number).
    pub winding_number: i32,
}

impl AnyonicHypergraph {
    pub fn new() -> Self {
        Self {
            node_stat: BTreeMap::new(),
            handovers: BTreeMap::new(),
            next_id: 1,
            winding_number: 0,
        }
    }

    /// Adiciona um nó com dada estatística.
    pub fn add_node(&mut self, node: String, alpha: AnyonStatistic) {
        self.node_stat.insert(node, alpha);
    }

    /// Atualiza a estatística de um nó (usado para Fallback/Annealing)
    pub fn update_node_stat(&mut self, node: &str, alpha: AnyonStatistic) {
        if self.node_stat.contains_key(node) {
            self.node_stat.insert(node.to_string(), alpha);
        }
    }

    /// Cria um handover entre dois nós. A estatística do handover é a média das dos nós.
    pub fn create_handover(
        &mut self,
        node_i: String,
        node_j: String,
        timestamp: u64,
        intensity: f64,
    ) -> Result<HandoverId, String> {
        let alpha_i = self.node_stat.get(&node_i).ok_or("Node i not found")?;
        let alpha_j = self.node_stat.get(&node_j).ok_or("Node j not found")?;
        // Média das estatísticas (simples média aritmética das frações)
        let avg_num = alpha_i.numerator * alpha_j.denominator + alpha_j.numerator * alpha_i.denominator;
        let avg_den = alpha_i.denominator * alpha_j.denominator * 2;
        // Simplificar fração (mdc)
        let gcd_val = gcd(avg_num, avg_den);
        let alpha = AnyonStatistic::new(avg_num / gcd_val, avg_den / gcd_val)
            .map_err(|_| "Invalid alpha")?;

        let id = HandoverId(self.next_id);
        self.next_id += 1;
        let handover = TopologicalHandover::new(id, node_i, node_j, alpha, timestamp, intensity);
        self.handovers.insert(id, handover);
        Ok(id)
    }

    /// Executa uma operação de braiding entre dois handovers identificados pelos seus IDs.
    pub fn braid(&mut self, id1: HandoverId, id2: HandoverId) -> Result<(), String> {
        // Verificar se existem
        if !self.handovers.contains_key(&id1) || !self.handovers.contains_key(&id2) {
            return Err("Handover not found".into());
        }
        // Precisamos de duas referências mutáveis – usar índices ou split_borrow.
        // Para simplificar, clonamos os dados e re-inserimos depois.
        let mut h1 = self.handovers.get(&id1).unwrap().clone();
        let mut h2 = self.handovers.get(&id2).unwrap().clone();

        let _phase = h1.braid_with(&mut h2);

        self.handovers.insert(id1, h1);
        self.handovers.insert(id2, h2);
        self.winding_number += 1; // incrementa invariante (simplificado)
        Ok(())
    }

    /// Calcula a coerência global do hipergrafo (produto de todas as fases acumuladas).
    pub fn global_coherence(&self) -> Complex64 {
        let mut product = Complex64::new(1.0, 0.0);
        for h in self.handovers.values() {
            product *= h.accumulated_phase;
        }
        product
    }

    /// Detecta vórtices anyónicos: nós cujo produto das fases dos handovers incidentes é não trivial.
    pub fn detect_vortices(&self) -> Vec<(String, Complex64)> {
        let mut node_phase: BTreeMap<String, Complex64> = BTreeMap::new();
        for h in self.handovers.values() {
            let entry_i = node_phase.entry(h.node_i.clone()).or_insert(Complex64::new(1.0, 0.0));
            *entry_i *= h.accumulated_phase;
            let entry_j = node_phase.entry(h.node_j.clone()).or_insert(Complex64::new(1.0, 0.0));
            *entry_j *= h.accumulated_phase;
        }
        node_phase
            .into_iter()
            .filter(|(_, phase)| (phase - Complex64::new(1.0, 0.0)).norm() > 1e-10)
            .collect()
    }

    /// Expurga vórtices topológicos, resetando a fase de handovers incidentes.
    /// Retorna o número de nós que tiveram suas fases limpas.
    pub fn purge_vortices(&mut self) -> usize {
        let vortices = self.detect_vortices();
        let num_purged = vortices.len();
        for (node, _) in vortices {
            println!("🚨 [ANYON] Expurgo de Vórtice Topológico no nó: {}", node);
            // Resetar fase de todos os handovers incidentes a este nó
            for h in self.handovers.values_mut() {
                if h.node_i == node || h.node_j == node {
                    h.accumulated_phase = Complex64::new(1.0, 0.0);
                    h.braid_partners.clear();
                }
            }
        }
        num_purged
    }

    /// Retorna a dissipação total para um dado momento k, somando sobre handovers.
    pub fn total_dissipation(&self, k: f64) -> (f64, f64) {
        let mut d2 = 0.0;
        let mut d3 = 0.0;
        for h in self.handovers.values() {
            d2 += h.dissipation(k, 2);
            d3 += h.dissipation(k, 3);
        }
        (d2, d3)
    }
}

/// Máximo divisor comum (Euclides)
fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anyon_statistic() {
        let a = AnyonStatistic::new(1, 3).unwrap();
        assert!((a.as_f64() - 1.0/3.0).abs() < 1e-10);
        assert!(a.is_anyonic());
        assert!(!a.is_bosonic());
        assert!(!a.is_fermionic());

        let phase = a.braiding_phase();
        assert!((phase - Complex64::from_polar(1.0, PI/3.0)).norm() < 1e-10);
    }

    #[test]
    fn test_braiding() {
        let mut graph = AnyonicHypergraph::new();
        graph.add_node("A".into(), AnyonStatistic::new(0, 1).unwrap()); // bosão
        graph.add_node("B".into(), AnyonStatistic::new(1, 3).unwrap());
        graph.add_node("C".into(), AnyonStatistic::new(2, 3).unwrap());

        let h1 = graph.create_handover("A".into(), "B".into(), 100, 1.0).unwrap();
        let h2 = graph.create_handover("B".into(), "C".into(), 200, 1.0).unwrap();

        let global_before = graph.global_coherence();
        graph.braid(h1, h2).unwrap();
        let global_after = graph.global_coherence();

        // O produto global deve permanecer 1 (conservação da fase total)
        assert!((global_after - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        // O invariante global (produto) não muda, mas a distribuição sim.
        assert_eq!(global_before, global_after);

        let vortices = graph.detect_vortices();
        assert_eq!(vortices.len(), 2); // B e C devem ter fase não trivial
    }
}
