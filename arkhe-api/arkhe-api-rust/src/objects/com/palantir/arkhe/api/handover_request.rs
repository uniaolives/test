#[derive(
    Debug,
    Clone,
    conjure_object::serde::Serialize,
    conjure_object::serde::Deserialize,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash
)]
#[serde(crate = "conjure_object::serde")]
#[conjure_object::private::staged_builder::staged_builder]
#[builder(crate = conjure_object::private::staged_builder, update, inline)]
pub struct HandoverRequest {
    #[serde(rename = "id")]
    id: i32,
    #[builder(into)]
    #[serde(rename = "targetNode")]
    target_node: String,
    #[builder(into)]
    #[serde(rename = "contextSummary")]
    context_summary: String,
    #[builder(into)]
    #[serde(rename = "priority")]
    priority: String,
}
impl HandoverRequest {
    #[inline]
    pub fn id(&self) -> i32 {
        self.id
    }
    #[inline]
    pub fn target_node(&self) -> &str {
        &*self.target_node
    }
    #[inline]
    pub fn context_summary(&self) -> &str {
        &*self.context_summary
    }
    #[inline]
    pub fn priority(&self) -> &str {
        &*self.priority
    }
}
