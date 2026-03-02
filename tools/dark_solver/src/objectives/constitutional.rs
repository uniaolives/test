#[cfg(feature = "z3-enabled")]
use crate::objectives::{Objective, ObjectiveResult};
#[cfg(feature = "z3-enabled")]
use z3::ast::{Ast, BV, Bool};
#[cfg(feature = "z3-enabled")]
use z3::{Config, Context, Solver};

pub struct SovereigntyObjective {
    pub node_id: u64,
    pub caller_id: u64,
    pub pre_stake: u64,
    pub post_stake: u64,
}

#[cfg(feature = "z3-enabled")]
impl Objective for SovereigntyObjective {
    fn evaluate(&self) -> ObjectiveResult {
        let cfg = Config::new();
        let ctx = Context::new(&cfg);
        let solver = Solver::new(&ctx);

        let node = BV::from_u64(&ctx, self.node_id, 64);
        let caller = BV::from_u64(&ctx, self.caller_id, 64);
        let pre = BV::from_u64(&ctx, self.pre_stake, 256);
        let post = BV::from_u64(&ctx, self.post_stake, 256);

        let caller_eq_node = caller._eq(&node);
        let stake_decreased = post.bvult(&pre);

        let property = stake_decreased.implies(&caller_eq_node);

        solver.assert(&Bool::not(&property));

        match solver.check() {
            z3::SatResult::Unsat => ObjectiveResult::Safe,
            z3::SatResult::Sat => ObjectiveResult::Violation(format!("{:?}", solver.get_model())),
            _ => ObjectiveResult::Unknown,
        }
    }
}
