use std::str::FromStr;
use anyhow::{Result, anyhow};

#[derive(Debug, Clone, PartialEq)]
pub enum Uqi {
    Classical(ClassicalUri),
    Superposed(StateVector),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClassicalUri {
    pub host: String,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StateVector {
    pub host: String,
    pub states: Vec<Eigenstate>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Eigenstate {
    pub amplitude: f64,
    pub uri: ClassicalUri,
}

impl FromStr for Uqi {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        if s.starts_with("timeline://") {
            let parts: Vec<&str> = s[11..].splitn(2, '/').collect();
            if parts.len() < 2 {
                return Err(anyhow!("Invalid timeline URI"));
            }

            // Basic query param parsing for t= and lambda=
            let path_parts: Vec<&str> = parts[1].splitn(2, '?').collect();
            let path = format!("/{}", path_parts[0]);

            Ok(Uqi::Classical(ClassicalUri {
                host: parts[0].to_string(),
                path,
            }))
        } else if s.starts_with("superposition://") {
            // format: superposition://host{amp:timeline://host/path|...}
            let brace_start = s.find('{').ok_or_else(|| anyhow!("Missing state vector"))?;
            let brace_end = s.find('}').ok_or_else(|| anyhow!("Unclosed state vector"))?;

            let host = &s[16..brace_start];
            let inner = &s[brace_start + 1..brace_end];

            let mut states = Vec::new();
            for part in inner.split('|') {
                let eigen_parts: Vec<&str> = part.splitn(2, ':').collect();
                if eigen_parts.len() < 2 {
                    return Err(anyhow!("Invalid eigenstate format"));
                }
                let amplitude = eigen_parts[0].trim().parse::<f64>()?;
                let uri_str = eigen_parts[1].trim();

                if let Uqi::Classical(uri) = Uqi::from_str(uri_str)? {
                    states.push(Eigenstate { amplitude, uri });
                } else {
                    return Err(anyhow!("Nested superpositions not supported in UQI v1"));
                }
            }

            Ok(Uqi::Superposed(StateVector {
                host: host.to_string(),
                states,
            }))
        } else {
            Err(anyhow!("Unknown UQI scheme"))
        }
    }
}
