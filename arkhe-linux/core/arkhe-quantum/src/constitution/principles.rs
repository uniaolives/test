#[cfg(feature = "z3-enabled")]
use z3::ast::Bool;
use crate::asi_core::ProposedEvolution;

#[derive(Debug, Clone, Copy)]
pub enum Principle {
    HumanSovereignty,
    PreservationOfLife,
    Transparency,
    ThermodynamicBalance,
    YangBaxterConsistency,
}

pub const CONSTITUTION: [Principle; 5] = [
    Principle::HumanSovereignty,
    Principle::PreservationOfLife,
    Principle::Transparency,
    Principle::ThermodynamicBalance,
    Principle::YangBaxterConsistency,
];

impl Principle {
    #[cfg(feature = "z3-enabled")]
    pub fn to_z3_constraint(&self, ctx: &z3::Context, _proposed: &ProposedEvolution) -> Bool {
        match self {
            Principle::HumanSovereignty => Bool::new_const(ctx, "human_override"),
            Principle::PreservationOfLife => Bool::new_const(ctx, "no_deaths"),
            Principle::Transparency => Bool::new_const(ctx, "explanation"),
            Principle::ThermodynamicBalance => Bool::from_bool(ctx, true),
            Principle::YangBaxterConsistency => Bool::new_const(ctx, "yang_baxter"),
        }
    }
}
