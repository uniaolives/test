use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FoundryObject {
    pub rid: String,
    pub properties: HashMap<String, serde_json::Value>,
}
