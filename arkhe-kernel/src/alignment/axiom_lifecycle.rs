use uuid::Uuid;
use crate::alignment::axiom_engine::Axiom;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AxiomStatus {
    Proposed,
    Voting,
    Ratified,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AxiomProposal {
    pub axiom: Axiom,
    pub status: AxiomStatus,
    pub votes: Vec<Vote>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Vote {
    pub node_id: Uuid,
    pub support: f64, // 0.0 to 1.0
    pub coherence: f64, // λ₂ of voter at time of vote
}

impl AxiomProposal {
    /// Tally votes weighted by coherence
    pub fn tally(&self) -> f64 {
        if self.votes.is_empty() {
            return 0.0;
        }

        let total_weight: f64 = self.votes.iter()
            .map(|v| v.coherence * v.support)
            .sum();

        let max_weight: f64 = self.votes.iter()
            .map(|v| v.coherence)
            .sum();

        if max_weight == 0.0 {
            return 0.0;
        }

        total_weight / max_weight
    }

    /// Check if ratified (requires 2/3 weighted majority)
    pub fn is_ratified(&self) -> bool {
        self.tally() > 0.667
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;
    use crate::alignment::axiom_engine::{Axiom, hash_content};

    #[test]
    fn test_axiom_voting() {
        let axiom = Axiom {
            id: Uuid::new_v4(),
            content: "Test".to_string(),
            domain: "Ethics".to_string(),
            hash: hash_content("Test"),
        };

        let mut proposal = AxiomProposal {
            axiom,
            status: AxiomStatus::Voting,
            votes: vec![
                Vote { node_id: Uuid::new_v4(), support: 1.0, coherence: 1.0 },
                Vote { node_id: Uuid::new_v4(), support: 0.0, coherence: 0.5 },
            ],
        };

        // Tally: (1.0*1.0 + 0.0*0.5) / (1.0 + 0.5) = 1.0 / 1.5 = 0.666...
        assert!(proposal.tally() < 0.667);
        assert!(!proposal.is_ratified());

        proposal.votes.push(Vote { node_id: Uuid::new_v4(), support: 1.0, coherence: 1.0 });
        // Tally: (1.0 + 0.0 + 1.0) / (1.0 + 0.5 + 1.0) = 2.0 / 2.5 = 0.8
        assert!(proposal.tally() > 0.667);
        assert!(proposal.is_ratified());
    }
}
