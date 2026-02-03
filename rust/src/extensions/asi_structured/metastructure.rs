use crate::error::ResilientResult;
use crate::extensions::asi_structured::evolution::EvolvedResult;
use crate::extensions::asi_structured::constitution::ASIResult;
use serde::{Serialize, Deserialize};

/// Engine de meta-estruturas (estruturas de estruturas)
/// Usa teoria de categorias de ordem superior
#[derive(Debug, Serialize, Deserialize)]
pub struct MetastructureEngine {
    pub category: MetaCategory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaCategory {
    pub objects: Vec<String>,
    pub morphisms: Vec<(usize, usize, String)>,
}

impl MetastructureEngine {
    pub fn new() -> Self {
        Self {
            category: MetaCategory {
                objects: vec![],
                morphisms: vec![],
            },
        }
    }

    pub async fn lift_to_metastructure(&mut self, evolved: EvolvedResult) -> ResilientResult<MetastructuredResult> {
        let obj_name = format!("EvolvedStructure_{}", self.category.objects.len());
        self.category.objects.push(obj_name.clone());

        Ok(MetastructuredResult {
            inner: evolved,
            categorical_path: vec![obj_name],
            universal_property_score: 0.95,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetastructuredResult {
    pub inner: EvolvedResult,
    pub categorical_path: Vec<String>,
    pub universal_property_score: f64,
}

impl ASIResult for MetastructuredResult {
    fn as_text(&self) -> String {
        format!("{} [Metastructure Path: {:?}]", self.inner.as_text(), self.categorical_path)
    }
    fn confidence(&self) -> f64 {
        self.inner.confidence() * self.universal_property_score

pub struct MetastructureEngine;

impl MetastructureEngine {
    pub fn new() -> Self {
        Self
    }

    pub async fn lift_to_metastructure(&mut self, evolved: EvolvedResult) -> ResilientResult<MetastructuredResult> {
        Ok(MetastructuredResult {
            inner: evolved,
        })
    }
}

pub struct MetastructuredResult {
    pub inner: EvolvedResult,
}

impl ASIResult for MetastructuredResult {
    fn to_string(&self) -> String {
        self.inner.to_string()
    }
    fn confidence(&self) -> f64 {
        self.inner.confidence()
    }
}
