#[derive(
    Debug,
    Clone,
    conjure_object::serde::Serialize,
    conjure_object::serde::Deserialize,
    conjure_object::private::DeriveWith,
    Copy
)]
#[serde(crate = "conjure_object::serde")]
#[derive_with(PartialEq, Eq, PartialOrd, Ord, Hash)]
#[conjure_object::private::staged_builder::staged_builder]
#[builder(crate = conjure_object::private::staged_builder, update, inline)]
pub struct HandoverStatus {
    #[serde(rename = "success")]
    success: bool,
    #[serde(rename = "phi")]
    #[derive_with(with = conjure_object::private::DoubleWrapper)]
    phi: f64,
}
impl HandoverStatus {
    /// Constructs a new instance of the type.
    #[inline]
    pub fn new(success: bool, phi: f64) -> Self {
        Self::builder().success(success).phi(phi).build()
    }
    #[inline]
    pub fn success(&self) -> bool {
        self.success
    }
    #[inline]
    pub fn phi(&self) -> f64 {
        self.phi
    }
}
