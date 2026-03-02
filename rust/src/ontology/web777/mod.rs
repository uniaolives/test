// rust/src/ontology/web777/mod.rs
pub mod syntax_mapping;
pub mod geometric_constraints;
pub mod semantic_query;

use std::collections::HashMap;
use petgraph::stable_graph::StableDiGraph;
use petgraph::graph::NodeIndex;
use serde::{Deserialize, Serialize};

pub use syntax_mapping::{SyntaxMapper, SyntaxFormat};
pub use geometric_constraints::{GeomStore, Geometry, ConstraintId};
pub use semantic_query::{Query, QueryResult, QueryError, SemanticQueryEngine};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyNode {
    pub id: String,
    pub label: Option<String>,
    pub attrs: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation(pub String);

pub struct Web777Engine {
    pub graph: StableDiGraph<OntologyNode, Relation>,
    pub index: HashMap<String, NodeIndex>,
    pub geom_store: GeomStore,
    pub syntax_mapper: SyntaxMapper,
    pub semantic_query: SemanticQueryEngine,
}

impl Web777Engine {
    pub fn new() -> Self {
        Self {
            graph: StableDiGraph::new(),
            index: HashMap::new(),
            geom_store: GeomStore::new(),
            syntax_mapper: SyntaxMapper::new(),
            semantic_query: SemanticQueryEngine::new(),
        }
    }

    pub fn upsert_node(&mut self, node: OntologyNode) -> NodeIndex {
        if let Some(&idx) = self.index.get(&node.id) {
            self.graph[idx] = node.clone();
            idx
        } else {
            let idx = self.graph.add_node(node.clone());
            self.index.insert(node.id.clone(), idx);
            idx
        }
    }

    pub fn add_relation(&mut self, src_id: &str, dst_id: &str, rel: &str) -> Result<(), String> {
        let src = self.index.get(src_id).ok_or_else(|| format!("source node '{}' not found", src_id))?;
        let dst = self.index.get(dst_id).ok_or_else(|| format!("target node '{}' not found", dst_id))?;
        self.graph.add_edge(*src, *dst, Relation(rel.to_string()));
        Ok(())
    }

    pub fn attach_geometry(&mut self, node_id: &str, geometry: Geometry) -> Result<ConstraintId, String> {
        let idx = self.index.get(node_id).ok_or_else(|| format!("node '{}' not found", node_id))?;
        self.geom_store.insert(*idx, geometry)
    }

    pub fn geometries_of(&self, node_id: &str) -> Option<Vec<(ConstraintId, &Geometry)>> {
        self.index.get(node_id).and_then(|idx| self.geom_store.get_all(*idx))
    }

    pub fn serialize(&self) -> Vec<u8> {
        // Mock serialization
        vec![]
    }

    pub fn deserialize(&mut self, _data: &[u8]) -> Result<(), String> {
        // Mock deserialization
        Ok(())
    }
}
