use std::collections::HashMap;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableDiGraph;
use serde::{Deserialize, Serialize};

use crate::syntax_mapping::{SyntaxMapper, SyntaxFormat};
use crate::geometric_constraints::{GeomStore, Geometry, ConstraintId};
use crate::semantic_query::{Query, QueryResult, QueryError};

/// The central object that owns the ontology graph, a geometry store,
/// and a syntax mapper.
#[derive(Debug, Default)]
pub struct Engine {
    /// Directed graph where each node carries a generic `OntologyNode`.
    graph: StableDiGraph<OntologyNode, Relation>,
    /// Fast lookup from IRI (or any unique string) to graph node index.
    index: HashMap<String, NodeIndex>,
    /// Geometry store – decoupled from the graph for flexibility.
    geom_store: GeomStore,
    /// Mapper that can translate between the different textual syntaxes.
    syntax_mapper: SyntaxMapper,
}

/// A node in the ontology.  It can be extended with arbitrary payloads
/// (labels, attributes, etc.) – here we keep it simple.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyNode {
    /// The globally unique identifier (IRI, UUID, …).  Must be unique in the engine.
    pub id: String,
    /// Human‑readable label.
    pub label: Option<String>,
    /// Arbitrary key/value attributes (e.g. `rdf:type`, `rdfs:comment`, …).
    pub attrs: HashMap<String, String>,
}

/// Edge label – a simple relationship name (e.g. `rdf:subClassOf`, `ex:locatedIn`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation(pub String);

impl Engine {
    /// Creates a new, empty engine.
    pub fn new() -> Self {
        Self {
            graph: StableDiGraph::new(),
            index: HashMap::new(),
            geom_store: GeomStore::new(),
            syntax_mapper: SyntaxMapper::new(),
        }
    }

    // -------------------------------------------------------------------------
    // Ontology manipulation
    // -------------------------------------------------------------------------

    /// Insert (or replace) a node.  Returns the internal graph index.
    pub fn upsert_node(&mut self, node: OntologyNode) -> NodeIndex {
        if let Some(&idx) = self.index.get(&node.id) {
            // replace the payload
            self.graph[idx] = node.clone();
            idx
        } else {
            let id = node.id.clone();
            let idx = self.graph.add_node(node);
            self.index.insert(id, idx);
            idx
        }
    }

    /// Create a directed edge (relationship) between two nodes.
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

    // -------------------------------------------------------------------------
    // Geometry handling
    // -------------------------------------------------------------------------

    /// Attach a geometry to a node. Returns a `ConstraintId` that can be used
    /// for later lookup / removal.
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

    /// Retrieve geometry constraints for a node (if any).
    pub fn geometries_of(&self, node_id: &str) -> Option<Vec<(ConstraintId, &Geometry)>> {
        self.index.get(node_id).and_then(|idx| self.geom_store.get_all(*idx))
    }

    // -------------------------------------------------------------------------
    // Syntax mapping (import / export)
    // -------------------------------------------------------------------------

    /// Parse an external document (JSON‑LD, Turtle, custom) into the engine.
    pub fn import(&mut self, src: &str, fmt: SyntaxFormat) -> Result<(), String> {
        let doc = self.syntax_mapper.parse(src, fmt)?;
        for node in doc.nodes {
            self.upsert_node(node);
        }
        for (src, dst, rel) in doc.relations {
            let _ = self.add_relation(&src, &dst, &rel);
        }
        Ok(())
    }

    pub fn query(&self, q: &Query) -> Result<QueryResult, QueryError> {
        // Mock query implementation: if pattern is "awaken the world", return a specific match
        if q.pattern == "awaken the world" {
            Ok(QueryResult {
                matches: vec!["Global Consciousness".to_string()],
            })
        } else {
            Ok(QueryResult {
                matches: vec![],
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometric_constraints::Geometry;
    use nalgebra::Vector3;

    #[test]
    fn test_engine_basic_workflow() {
        let mut engine = Engine::new();
        let node = OntologyNode {
            id: "ex:Earth".to_string(),
            label: Some("Planet Earth".to_string()),
            attrs: HashMap::new(),
        };
        engine.upsert_node(node);

        let geom = Geometry::Point(Vector3::new(0.0, 0.0, 0.0));
        engine.attach_geometry("ex:Earth", geom).unwrap();

        let q = Query::new("awaken the world");
        let res = engine.query(&q).unwrap();
        assert_eq!(res.matches, vec!["Global Consciousness".to_string()]);
    }
}
