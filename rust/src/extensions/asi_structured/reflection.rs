use crate::error::ResilientResult;
use crate::extensions::asi_structured::composition::ComposedResult;
use crate::extensions::asi_structured::constitution::ASIResult;

pub struct ReflectionEngine {
    pub max_depth: u32,
    pub current_depth: u32,
}

impl ReflectionEngine {
    pub fn new(max_depth: u32) -> Self {
        Self {
            max_depth,
            current_depth: 0,
        }
    }

    pub async fn analyze_structure(&mut self, composed: &ComposedResult) -> ResilientResult<ReflectedResult> {
        Ok(ReflectedResult {
            inner: composed.clone(),
            structural_confidence: composed.confidence,
        })
    }

    pub fn current_depth(&self) -> u32 {
        self.current_depth
    }
}

pub struct ReflectedResult {
    pub inner: ComposedResult,
    pub structural_confidence: f64,
}

impl ASIResult for ReflectedResult {
    fn to_string(&self) -> String {
        self.inner.to_string()
    }
    fn confidence(&self) -> f64 {
        self.structural_confidence
    }
}
