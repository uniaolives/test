#[derive(
    Debug,
    Clone,
    conjure_object::serde::Serialize,
    conjure_object::serde::Deserialize,
    conjure_object::private::DeriveWith
)]
#[serde(crate = "conjure_object::serde")]
#[derive_with(PartialEq, Eq, PartialOrd, Ord, Hash)]
#[conjure_object::private::staged_builder::staged_builder]
#[builder(crate = conjure_object::private::staged_builder, update, inline)]
pub struct HandoverResponse {
    #[builder(into)]
    #[serde(rename = "status")]
    status: String,
    #[builder(into)]
    #[serde(rename = "handoverId")]
    handover_id: String,
    #[serde(rename = "phiRemote")]
    #[derive_with(with = conjure_object::private::DoubleWrapper)]
    phi_remote: f64,
}
impl HandoverResponse {
    /// Constructs a new instance of the type.
    #[inline]
    pub fn new(
        status: impl Into<String>,
        handover_id: impl Into<String>,
        phi_remote: f64,
    ) -> Self {
        Self::builder()
            .status(status)
            .handover_id(handover_id)
            .phi_remote(phi_remote)
            .build()
    }
    #[inline]
    pub fn status(&self) -> &str {
        &*self.status
    }
    #[inline]
    pub fn handover_id(&self) -> &str {
        &*self.handover_id
    }
    #[inline]
    pub fn phi_remote(&self) -> f64 {
        self.phi_remote
    }
}
