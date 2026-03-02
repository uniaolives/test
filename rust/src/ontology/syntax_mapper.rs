// rust/src/ontology/syntax_mapper.rs
// SASC v70.0: Universal Syntax Mapping Matrix

use std::collections::HashMap;

pub struct UniversalSyntaxMapper {
    pub mapping_table: HashMap<String, OntologicalPrimitive>,
}

#[derive(Debug, Clone)]
pub enum OntologicalPrimitive {
    State {
        physical_type: PhysicalType,
        conservation: Conservation,
        topology: Topology,
    },
    Transform {
        input: Vec<String>,
        output: String,
        entropy_cost: String,
        geometric_path: String,
    },
    Branch {
        paths: u32,
        condition: String,
        selection: String,
    },
    Cycle {
        period: String,
        stability: String,
        winding_number: String,
    },
    Manifold {
        dimension: u32,
        metric: String,
        curvature: String,
    },
    Entity {
        properties: Vec<String>,
        relationships: Vec<String>,
        constraints: Vec<String>,
    },
    Decoherence {
        cause: String,
        recovery: String,
        entropy_production: String,
    },
    Entanglement {
        degree: String,
        coherence: String,
        distance: f64,
    },
    ParallelTransport {
        connection: String,
        holonomy: String,
    },
}

#[derive(Debug, Clone)]
pub enum PhysicalType { Information, Energy, Coherence, Sigma }

#[derive(Debug, Clone)]
pub enum Conservation { Energy, Information, None }

#[derive(Debug, Clone)]
pub enum Topology { PointInSpace, Torus, Manifold }

impl UniversalSyntaxMapper {
    pub fn new() -> Self {
        let mut mapping_table = HashMap::new();

        mapping_table.insert("variable".to_string(), OntologicalPrimitive::State {
            physical_type: PhysicalType::Information,
            conservation: Conservation::Energy,
            topology: Topology::PointInSpace,
        });

        mapping_table.insert("function".to_string(), OntologicalPrimitive::Transform {
            input: vec!["State".to_string()],
            output: "State".to_string(),
            entropy_cost: "Calculated".to_string(),
            geometric_path: "Geodesic".to_string(),
        });

        Self { mapping_table }
    }

    pub fn map_all_languages(&self) -> HashMap<String, OntologicalPrimitive> {
        self.mapping_table.clone()
    }
}
