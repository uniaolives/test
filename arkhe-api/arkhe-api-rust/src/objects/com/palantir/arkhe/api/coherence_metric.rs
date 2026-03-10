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
pub struct CoherenceMetric {
    #[serde(rename = "globalR")]
    #[derive_with(with = conjure_object::private::DoubleWrapper)]
    global_r: f64,
    #[serde(rename = "nodeLambda")]
    #[derive_with(with = conjure_object::private::DoubleWrapper)]
    node_lambda: f64,
    #[serde(rename = "stable")]
    stable: bool,
}
impl CoherenceMetric {
    /// Constructs a new instance of the type.
    #[inline]
    pub fn new(global_r: f64, node_lambda: f64, stable: bool) -> Self {
        Self::builder()
            .global_r(global_r)
            .node_lambda(node_lambda)
            .stable(stable)
            .build()
    }
    #[inline]
    pub fn global_r(&self) -> f64 {
        self.global_r
    }
    #[inline]
    pub fn node_lambda(&self) -> f64 {
        self.node_lambda
    }
    #[inline]
    pub fn stable(&self) -> bool {
        self.stable
    }
}
