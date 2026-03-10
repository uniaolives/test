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
pub struct AgentCard {
    #[builder(into)]
    #[serde(rename = "name")]
    name: String,
    #[builder(default, list(item(type = String, into)))]
    #[serde(rename = "skills", skip_serializing_if = "Vec::is_empty", default)]
    skills: Vec<String>,
    #[builder(into)]
    #[serde(rename = "endpoint")]
    endpoint: String,
    #[builder(default, list(item(type = String, into)))]
    #[serde(rename = "identityTraces", skip_serializing_if = "Vec::is_empty", default)]
    identity_traces: Vec<String>,
    #[builder(into)]
    #[serde(rename = "authRequirements")]
    auth_requirements: String,
}
impl AgentCard {
    /// Constructs a new instance of the type.
    #[inline]
    pub fn new(
        name: impl Into<String>,
        endpoint: impl Into<String>,
        auth_requirements: impl Into<String>,
    ) -> Self {
        Self::builder()
            .name(name)
            .endpoint(endpoint)
            .auth_requirements(auth_requirements)
            .build()
    }
    #[inline]
    pub fn name(&self) -> &str {
        &*self.name
    }
    #[inline]
    pub fn skills(&self) -> &[String] {
        &*self.skills
    }
    #[inline]
    pub fn endpoint(&self) -> &str {
        &*self.endpoint
    }
    #[inline]
    pub fn identity_traces(&self) -> &[String] {
        &*self.identity_traces
    }
    #[inline]
    pub fn auth_requirements(&self) -> &str {
        &*self.auth_requirements
    }
}
