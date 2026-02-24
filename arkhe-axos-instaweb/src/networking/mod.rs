pub mod quantum_hybrid;
pub mod vectorized_v2;

pub struct InstawebNode;
impl InstawebNode {
    pub fn with_constitution() -> Self { Self }
    pub async fn route(&self, _task: &crate::Task) -> Result<crate::Path, crate::execution::Error> {
        Ok(crate::Path)
    }
}
