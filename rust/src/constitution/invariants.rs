// rust/src/constitution/invariants.rs
use crate::state::ResilientState;
use regex::Regex;
use lazy_static::lazy_static;

pub struct CGEInvariant {
    pub id: String,
    pub name: String,
    pub description: String,
    pub validator: Box<dyn Fn(&ResilientState) -> InvariantResult + Send + Sync>,
}

pub struct InvariantResult {
    pub name: String,
    pub passed: bool,
    pub details: String,
}

pub struct InvariantSet {
    invariants: Vec<CGEInvariant>,
}

impl Default for InvariantSet {
    fn default() -> Self {
        let mut set = Self { invariants: Vec::new() };
        set.register_default_invariants();
        set
    }
}

impl InvariantSet {
    fn register_default_invariants(&mut self) {
        self.invariants.extend(vec![
            CGEInvariant {
                id: "C1".to_string(),
                name: "No Secrets in State".to_string(),
                description: "State must not contain API keys, private keys, or other secrets".to_string(),
                validator: Box::new(|state: &ResilientState| {
                    let mut failed_secrets = Vec::new();
                    if contains_secrets(&state.memory.summary) {
                        failed_secrets.push("memory.summary");
                    }
                    InvariantResult {
                        name: "C1".to_string(),
                        passed: failed_secrets.is_empty(),
                        details: if failed_secrets.is_empty() {
                            "OK".to_string()
                        } else {
                            format!("Secrets found in: {}", failed_secrets.join(", "))
                        },
                    }
                }),
            },
            CGEInvariant {
                id: "C2".to_string(),
                name: "Monotonic Temporal Progress".to_string(),
                description: "State timestamps must progress forward".to_string(),
                validator: Box::new(|state: &ResilientState| {
                    let mut issues = Vec::new();
                    if state.created_at > state.updated_at {
                        issues.push("created_at > updated_at");
                    }
                    InvariantResult {
                        name: "C2".to_string(),
                        passed: issues.is_empty(),
                        details: if issues.is_empty() {
                            "OK".to_string()
                        } else {
                            format!("Temporal issues: {}", issues.join(", "))
                        },
                    }
                }),
            },
            CGEInvariant {
                id: "C3".to_string(),
                name: "State Size Bounds".to_string(),
                description: "State must be under size limit for economic persistence".to_string(),
                validator: Box::new(|state: &ResilientState| {
                    match state.estimate_size() {
                        Ok(size) => {
                            let limit = 102_400;
                            InvariantResult {
                                name: "C3".to_string(),
                                passed: size <= limit,
                                details: if size <= limit {
                                    format!("OK ({} bytes)", size)
                                } else {
                                    format!("Exceeds limit: {} > {} bytes", size, limit)
                                },
                            }
                        }
                        Err(e) => InvariantResult {
                            name: "C3".to_string(),
                            passed: false,
                            details: format!("Failed to estimate size: {}", e),
                        },
                    }
                }),
            },
        ]);
    }

    pub fn check_all(&self, state: &ResilientState) -> Vec<InvariantResult> {
        self.invariants.iter()
            .map(|invariant| {
                let validator = &invariant.validator;
                validator(state)
            })
            .collect()
    }
}

fn contains_secrets(text: &str) -> bool {
    lazy_static! {
        static ref SECRET_PATTERNS: Vec<Regex> = vec![
            Regex::new(r#"(?i)api[_-]?key\s*[:=]\s*['"]?[a-zA-Z0-9_-]{20,}['"]?"#).unwrap(),
            Regex::new(r#"(?i)secret\s*[:=]\s*['"]?[a-zA-Z0-9_-]{20,}['"]?"#).unwrap(),
            Regex::new(r#"(?i)private[_-]?key\s*[:=]\s*['"]?-----BEGIN.*END-----['"]?"#).unwrap(),
            Regex::new(r#"(?i)password\s*[:=]\s*['"]?.{8,}['"]?"#).unwrap(),
            Regex::new(r#"[\w\.-]+@[\w\.-]+\.\w+"#).unwrap(), // Email
            Regex::new(r#"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"#).unwrap(), // IP
        ];
    }

    SECRET_PATTERNS.iter().any(|pattern| pattern.is_match(text))
}
