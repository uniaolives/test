use crate::types::Provider;
use std::collections::{HashMap, HashSet};

pub const HHI_THRESHOLD: f64 = 1800.0;
pub const MARKET_SHARE_THRESHOLD: f64 = 0.25;
pub const MIN_REDUNDANCY: usize = 3;

pub struct MarketConcentrationMonitor {
    pub providers: Vec<Provider>,
}

impl MarketConcentrationMonitor {
    pub fn new(providers: Vec<Provider>) -> Self {
        Self { providers }
    }

    /// Calculates the Herfindahl-Hirschman Index (HHI)
    pub fn calculate_hhi(&self) -> f64 {
        self.providers
            .iter()
            .map(|p| (p.market_share * 100.0).powi(2))
            .sum()
    }

    pub fn check_antitrust_compliance(&self) -> bool {
        let hhi = self.calculate_hhi();
        if hhi > HHI_THRESHOLD {
            println!("ANTITRUST VIOLATION: HHI {} exceeds threshold {}", hhi, HHI_THRESHOLD);
            return false;
        }

        for provider in &self.providers {
            if provider.market_share > MARKET_SHARE_THRESHOLD {
                println!("ANTITRUST VIOLATION: Provider {} exceeds market share threshold", provider.id);
                return false;
            }
        }

        true
    }

    pub fn build_dependency_graph(&self) -> HashMap<String, Vec<String>> {
        let mut graph = HashMap::new();
        for p in &self.providers {
            graph.insert(p.id.clone(), p.dependencies.clone());
        }
        graph
    }

    pub fn find_critical_nodes(&self) -> HashSet<String> {
        let graph = self.build_dependency_graph();
        let mut all_deps = HashSet::new();
        for deps in graph.values() {
            for dep in deps {
                all_deps.insert(dep.clone());
            }
        }

        let mut critical = HashSet::new();
        for id in graph.keys() {
            if !all_deps.contains(id) {
                critical.insert(id.clone());
            }
        }
        critical
    }

    pub fn check_redundancy(&self) -> bool {
        let critical_nodes = self.find_critical_nodes();
        critical_nodes.len() >= MIN_REDUNDANCY
    }
}
