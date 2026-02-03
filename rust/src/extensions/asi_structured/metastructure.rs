use crate::error::ResilientResult;
use crate::extensions::asi_structured::evolution::EvolvedResult;
use crate::extensions::asi_structured::constitution::ASIResult;

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
