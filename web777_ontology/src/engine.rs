use std::collections::HashMap;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableDiGraph;
use serde::{Deserialize, Serialize};

use crate::syntax_mapping::{SyntaxMapper, SyntaxFormat};
use crate::geometric_constraints::{GeomStore, Geometry, ConstraintId};
use crate::semantic_query::{Query, QueryResult, QueryError};

/// The central object that owns the ontology graph, a geometry store,
/// and a syntax mapper.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Engine {
    /// Directed graph where each node carries a generic `OntologyNode`.
    pub graph: StableDiGraph<OntologyNode, Relation>,
    /// Fast lookup from IRI (or any unique string) to graph node index.
    pub index: HashMap<String, NodeIndex>,
    /// Geometry store – decoupled from the graph for flexibility.
    pub geom_store: GeomStore,
    /// Mapper that can translate between the different textual syntaxes.
    pub syntax_mapper: SyntaxMapper,
    /// Semantic query engine.
    pub semantic_query: crate::semantic_query::SemanticQueryEngine,
    /// Performance tuning engine (Oracle DBA).
    pub dba: OracleDBA,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct OracleDBA {
    pub statistics: DBStats,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct DBStats {
    pub query_count: u64,
    pub cache_hits: u64,
    pub execution_time_total_ms: u64,
}

/// A node in the ontology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyNode {
    pub id: String,
    pub label: Option<String>,
    pub attrs: HashMap<String, String>,
}

/// Edge label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation(pub String);

impl Engine {
    pub fn new() -> Self {
        Self {
            graph: StableDiGraph::new(),
            index: HashMap::new(),
            geom_store: GeomStore::new(),
            syntax_mapper: SyntaxMapper::new(),
            semantic_query: crate::semantic_query::SemanticQueryEngine::new(),
            dba: OracleDBA::default(),
        }
    }

    pub fn upsert_node(&mut self, node: OntologyNode) -> NodeIndex {
        if let Some(&idx) = self.index.get(&node.id) {
            self.graph[idx] = node.clone();
            idx
        } else {
            let id = node.id.clone();
            let idx = self.graph.add_node(node);
            self.index.insert(id, idx);
            idx
        }
    }

    pub fn add_relation(&mut self, src_id: &str, dst_id: &str, rel: &str) -> Result<(), String> {
        let src = self
            .index
            .get(src_id)
            .ok_or_else(|| format!("source node '{}' not found", src_id))?;
        let dst = self
            .index
            .get(dst_id)
            .ok_or_else(|| format!("target node '{}' not found", dst_id))?;
        self.graph.add_edge(*src, *dst, Relation(rel.to_string()));
        Ok(())
    }

    pub fn attach_geometry(
        &mut self,
        node_id: &str,
        geometry: Geometry,
    ) -> Result<ConstraintId, String> {
        let idx = self
            .index
            .get(node_id)
            .ok_or_else(|| format!("node '{}' not found", node_id))?;
        self.geom_store.insert(*idx, geometry)
    }

    pub fn geometries_of(&self, node_id: &str) -> Option<Vec<(ConstraintId, &Geometry)>> {
        self.index.get(node_id).and_then(|idx| self.geom_store.get_all(*idx))
    }

    pub fn query(&mut self, q: &crate::semantic_query::SemanticQuery) -> Result<Vec<QueryResult>, String> {
        let start = std::time::Instant::now();
        self.dba.statistics.query_count += 1;

        let nodes = self.index.keys().cloned();
        let results = self.semantic_query.execute(q, nodes)?;

        // Simulating DBA performance tracking
        self.dba.statistics.execution_time_total_ms += start.elapsed().as_millis() as u64;

        Ok(results)
    }

    pub fn serialize(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    pub fn deserialize(&mut self, data: &[u8]) -> Result<(), String> {
        let other: Engine = serde_json::from_slice(data).map_err(|e| e.to_string())?;
        *self = other;
        Ok(())
    }
}
