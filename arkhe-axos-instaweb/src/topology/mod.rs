pub struct HyperbolicManifold;

impl HyperbolicManifold {
    pub fn with_constitution() -> Self {
        Self
    }

    pub fn verify_invariants(&self, _state: &crate::dynamics::State) -> Result<(), crate::execution::Error> {
        Ok(())
    }
}
