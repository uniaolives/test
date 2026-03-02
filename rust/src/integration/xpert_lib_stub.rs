use crate::biology::*;
use crate::integration::xpert_omega_adapter::ProcessedInput;
use thiserror::Error;

pub struct XPertModel;

impl XPertModel {
    pub fn load_secure(_path: &str) -> Result<Self, crate::integration::xpert_omega_adapter::IntegrityError> {
        Ok(XPertModel)
    }

    pub fn predict(&self, _input: ProcessedInput) -> Result<GeneExpressionPrediction, XPertError> {
        Ok(vec![0.1, 0.2, 0.3])
    }
}

#[derive(Debug, Error)]
pub enum XPertError {
    #[error("Internal XPert error")]
    Internal,
}
