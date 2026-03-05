use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::ritual_sim::ARKHE_CONSTITUTION_HASH;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ProjectStatus {
    Proposed,
    Active,
    Completed,
    Halted,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Milestone {
    pub id: Uuid,
    pub description: String,
    pub amount: u64,
    pub achieved: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Project {
    pub id: Uuid,
    pub description: String,
    pub funding_required: u64,
    pub funding_received: u64,
    pub quantum_computing_hours: u64,
    pub status: ProjectStatus,
    pub milestones: Vec<Milestone>,
}

pub struct PhoenixContract {
    pub totem: String,
    pub projects: Vec<Project>,
}

impl PhoenixContract {
    pub fn new() -> Self {
        Self {
            totem: ARKHE_CONSTITUTION_HASH.to_string(),
            projects: Vec::new(),
        }
    }

    pub fn submit_proposal(&mut self, description: String, funding: u64) -> Uuid {
        let id = Uuid::new_v4();
        let project = Project {
            id,
            description,
            funding_required: funding,
            funding_received: 0,
            quantum_computing_hours: 0,
            status: ProjectStatus::Proposed,
            milestones: Vec::new(),
        };
        self.projects.push(project);
        id
    }

    pub fn allocate_funds(&mut self, project_id: Uuid, amount: u64) -> bool {
        if let Some(project) = self.projects.iter_mut().find(|p| p.id == project_id) {
            project.funding_received += amount;
            if project.funding_received >= project.funding_required {
                project.status = ProjectStatus::Active;
            }
            return true;
        }
        false
    }
}
