//! Public API of the Web777 Ontology Engine.
//! The crate is deliberately small – most of the logic lives in the
//! sub‑modules.

pub mod engine;
pub mod syntax_mapping;
pub mod geometric_constraints;
pub mod semantic_query;

pub use engine::{Engine, OntologyNode, Relation};
pub use syntax_mapping::{SyntaxMapper, SyntaxFormat};
pub use geometric_constraints::{GeomStore, Geometry, ConstraintId};
pub use semantic_query::{Query, QueryResult, QueryError};
