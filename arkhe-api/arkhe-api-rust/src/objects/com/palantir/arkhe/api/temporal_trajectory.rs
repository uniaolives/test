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
pub struct TemporalTrajectory {
    #[builder(default, list(item(type = f64)))]
    #[serde(rename = "points", skip_serializing_if = "Vec::is_empty", default)]
    #[derive_with(with = conjure_object::private::DoubleWrapper)]
    points: Vec<f64>,
    #[serde(rename = "curvature")]
    #[derive_with(with = conjure_object::private::DoubleWrapper)]
    curvature: f64,
}
impl TemporalTrajectory {
    /// Constructs a new instance of the type.
    #[inline]
    pub fn new(curvature: f64) -> Self {
        Self::builder().curvature(curvature).build()
    }
    #[inline]
    pub fn points(&self) -> &[f64] {
        &*self.points
    }
    #[inline]
    pub fn curvature(&self) -> f64 {
        self.curvature
    }
}
