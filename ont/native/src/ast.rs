#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Paradigm {
    Functional,
    Imperative,
    Agent,
    OO,
    Substrate,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OntoType {
    Pure(Box<OntoType>),
    Mutable(Box<OntoType>),
    Agent(Box<OntoType>),
    Substrate(Box<OntoType>),
pub enum Type {
    Pure(Box<Type>),
    Mutable(Box<Type>),
    Agent(Box<Type>),
    Substrate(Box<Type>),
    Object(String),
    Int,
    Float,
    Bool,
    String,
    Named(String, Vec<OntoType>),
    Named(String, Vec<Type>),
    Unit,
    Bytes,
    Address,
    TypeVar(String),
    Function(Vec<Type>, Box<Type>),
}

#[derive(Debug, Clone)]
pub struct TypeAnnotation {
    pub name: String,
    pub type_expr: Type,
    pub paradigm: Paradigm,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub paradigm: Paradigm,
    pub params: Vec<(String, OntoType)>,
    pub return_type: OntoType,
    pub params: Vec<(String, Type)>,
    pub return_type: Type,
    pub body: Body,
    pub constraints: Vec<Constraint>,
    pub target_paradigm: Option<Paradigm>,
}

#[derive(Debug, Clone)]
pub struct Body {
    pub content: String,
}

impl Body {
    pub fn to_solidity(&self) -> Result<String, crate::compiler::CompilerError> {
        Ok(format!("/* {} */", self.content))
        Ok(format!("// {} ", self.content))
    }
}

#[derive(Debug, Clone)]
pub struct Agent {
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub field_type: OntoType,
    pub field_type: Type,
}

#[derive(Debug, Clone)]
pub struct Method {
    pub name: String,
    pub params: Vec<(String, OntoType)>,
    pub return_type: OntoType,
    pub params: Vec<(String, Type)>,
    pub return_type: Type,
}

#[derive(Debug, Clone)]
pub struct Class {
    pub name: String,
    pub fields: Vec<Field>,
    pub methods: Vec<Method>,
}

#[derive(Debug, Clone)]
pub struct Transmutation {
    pub name: String,
    pub constraints: Vec<Constraint>,
}

#[derive(Debug, Clone)]
pub struct Constraint {
    pub name: String,
}

impl Constraint {
    pub fn parse_o_n_complexity(&self) -> Option<u64> {
        if self.name.starts_with("O(") {
            Some(100) // Mock value
        } else {
            None
        }
    }

    pub fn contains(&self, s: &str) -> bool {
        self.name.contains(s)
    }
}

pub struct OntologyProgram {
    pub functions: Vec<Function>,
    pub agents: Vec<Agent>,
    pub classes: Vec<Class>,
    pub transmutations: Vec<Transmutation>,
}

impl OntologyProgram {
    pub fn agents(&self) -> impl Iterator<Item = &Agent> {
        self.agents.iter()
    }

    pub fn classes(&self) -> impl Iterator<Item = &Class> {
        self.classes.iter()
    }

    pub fn transmutations(&self) -> Vec<&Transmutation> {
        self.transmutations.iter().collect()
    }
}
