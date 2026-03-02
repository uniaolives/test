// rust/src/babel/syntax.rs

pub struct NeoCode {
    pub syntax: String,
    pub example: String,
}

pub struct State {
    pub name: String,
    pub unit: String,
    pub value: f64,
}

pub struct Transform {
    pub name: String,
    pub entropy_cost: f64,
}

pub struct Invariant {
    pub name: String,
    pub properties: Vec<String>,
}

pub struct GeometricAST {
    pub manifolds: Vec<String>,
    pub constraints: Vec<String>,
    pub topologies: Vec<String>,
}

pub struct ConstrainedGeometry {
    pub closed_manifolds: Vec<String>,
    pub satisfied_constraints: Vec<String>,
    pub integer_winding: Vec<i32>,
}
