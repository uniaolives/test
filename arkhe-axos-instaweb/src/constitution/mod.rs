#[derive(Default)]
pub struct Constitution;

impl Constitution {
    pub fn verify(&self, _task: &crate::Task) -> Result<(), crate::execution::Error> {
        Ok(())
    }
}
