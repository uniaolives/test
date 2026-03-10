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
pub struct Orb {
    #[builder(into)]
    #[serde(rename = "id")]
    id: String,
    #[serde(rename = "lambda2")]
    #[derive_with(with = conjure_object::private::DoubleWrapper)]
    lambda2: f64,
    #[serde(rename = "phiQ")]
    #[derive_with(with = conjure_object::private::DoubleWrapper)]
    phi_q: f64,
    #[serde(rename = "hValue")]
    #[derive_with(with = conjure_object::private::DoubleWrapper)]
    h_value: f64,
    #[builder(into)]
    #[serde(rename = "timestamp")]
    timestamp: String,
}
impl Orb {
    #[inline]
    pub fn id(&self) -> &str {
        &*self.id
    }
    #[inline]
    pub fn lambda2(&self) -> f64 {
        self.lambda2
    }
    #[inline]
    pub fn phi_q(&self) -> f64 {
        self.phi_q
    }
    #[inline]
    pub fn h_value(&self) -> f64 {
        self.h_value
    }
    #[inline]
    pub fn timestamp(&self) -> &str {
        &*self.timestamp
    }
}
