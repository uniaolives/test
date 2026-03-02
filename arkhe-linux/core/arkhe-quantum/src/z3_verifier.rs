#[cfg(feature = "z3-enabled")]
use z3::{Config, Context, SatResult, Solver};
use crate::asi_core::ProposedEvolution;
use crate::constitution::principles::Principle;
use crate::KrausOperator;

#[derive(Debug)]
pub enum ConstitutionalViolation {
    Multiple,
    Unknown,
}

pub struct VerifiedAction {
    pub world_action: KrausOperator,
    pub self_modification: crate::self_modification::SelfModification,
}

impl VerifiedAction {
    pub fn from_proposed(proposed: ProposedEvolution) -> Self {
        VerifiedAction {
            world_action: proposed.world_action,
            self_modification: proposed.self_modification,
        }
    }
}

pub struct Z3Solver;

impl Z3Solver {
    pub fn project_to_constitutional_subspace(
        proposed: &ProposedEvolution,
        _constitution: &[Principle],
    ) -> Result<VerifiedAction, ConstitutionalViolation> {
        log::debug!("Invocando verificação constitucional...");

        #[cfg(feature = "z3-enabled")]
        {
            let cfg = Config::new();
            let ctx = Context::new(&cfg);
            let solver = Solver::new(&ctx);

            for principle in _constitution {
                let constraint = principle.to_z3_constraint(&ctx, proposed);
                solver.assert(&constraint);
            }

            let entropy_change = z3::ast::Real::from_real(&ctx, (proposed.expected_entropy_change * 1000.0) as i32, 1000);
            let max_change = z3::ast::Real::from_real(&ctx, 10, 1);
            solver.assert(&entropy_change.le(&max_change));

            match solver.check() {
                SatResult::Sat => {
                    log::info!("Z3: SAT – evolução é constitucional.");
                    Ok(VerifiedAction::from_proposed(proposed.clone()))
                }
                _ => Err(ConstitutionalViolation::Multiple),
            }
        }

        #[cfg(not(feature = "z3-enabled"))]
        {
            log::warn!("Z3 desativado. Usando verificação heurística (Mock).");
            if proposed.expected_entropy_change > 0.1 {
                return Err(ConstitutionalViolation::Multiple);
            }
            Ok(VerifiedAction::from_proposed(proposed.clone()))
        }
    }
}
