pub struct AxosKernel;

#[derive(Debug)]
pub enum Error {
    ConstitutionalViolation,
    IntegrationError,
    TopologyError,
}

impl AxosKernel {
    pub fn with_constitution() -> Self {
        Self
    }

    pub async fn integrate(&self, _task: crate::Task, _path: crate::Path) -> Result<crate::dynamics::State, Error> {
        Ok(crate::dynamics::State::default())
    }
}
