//! harmonia/src/soul/semantic_network.rs
//! Rede que conecta deep fakes, patentes, energia, renda básica, etc.

use std::collections::HashMap;
use petgraph::graph::{NodeIndex, DiGraph};
use petgraph::Direction;

#[derive(Debug, Clone)]
pub enum ContextDomain {
    Information,
    Law,
    Ecology,
    Energy,
    Economy,
    Deliberation,
}

#[derive(Debug, Clone)]
pub struct Concept {
    pub id: String,
    pub name: String,
    pub domain: ContextDomain,
    pub coords: (f64, f64, f64),
    pub stability: f64,
}

pub struct UnifiedSemanticNetwork {
    pub graph: DiGraph<Concept, String>,
    pub index_map: HashMap<String, NodeIndex>,
}

impl UnifiedSemanticNetwork {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            index_map: HashMap::new(),
        }
    }

    pub fn add_concept(&mut self, concept: Concept) {
        let id = concept.id.clone();
        let node_index = self.graph.add_node(concept);
        self.index_map.insert(id, node_index);
    }

    pub fn add_relation(&mut self, source_id: &str, target_id: &str, relation_type: &str) {
        if let (Some(&source), Some(&target)) = (self.index_map.get(source_id), self.index_map.get(target_id)) {
            self.graph.add_edge(source, target, relation_type.to_string());
        }
    }

    pub fn propagate_effect(&self, start_id: &str, strength: f64) -> HashMap<String, f64> {
        let mut effects = HashMap::new();
        let mut queue = Vec::new();

        if let Some(&start_node) = self.index_map.get(start_id) {
            queue.push((start_node, strength));
        }

        while let Some((current_node, current_strength)) = queue.pop() {
            if current_strength < 0.1 { continue; }

            let concept = &self.graph[current_node];
            let entry = effects.entry(concept.id.clone()).or_insert(0.0);
            *entry += current_strength;

            for neighbor in self.graph.neighbors_directed(current_node, Direction::Outgoing) {
                queue.push((neighbor, current_strength * 0.8));
            }
        }

        effects
    }
}

pub fn build_v2_context_network() -> UnifiedSemanticNetwork {
    let mut net = UnifiedSemanticNetwork::new();

    let concepts = vec![
        Concept { id: "deepfake".into(), name: "Deep Fakes".into(), domain: ContextDomain::Information, coords: (0.1, 0.9, 0.2), stability: 0.5 },
        Concept { id: "patent_break".into(), name: "Quebra de Patente Médica".into(), domain: ContextDomain::Law, coords: (0.7, 0.6, 0.8), stability: 0.5 },
        Concept { id: "ecological_health".into(), name: "Saúde Ecológica".into(), domain: ContextDomain::Ecology, coords: (0.2, 0.1, 0.7), stability: 0.5 },
    ];

    for c in concepts {
        net.add_concept(c);
    }

    net.add_relation("deepfake", "patent_break", "undermines_trust");
    net.add_relation("patent_break", "ecological_health", "supports_public_health");

    net
}
