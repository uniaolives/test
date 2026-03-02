// ==============================================
// ONTOLOGY STANDARD LIBRARY v0.6.0
// ==============================================

pub mod storage;
pub mod crypto;
pub mod events;
pub mod math;

use crate::type_checker::type_env::TypeEnv;

pub fn register_std(env: &mut TypeEnv) {
    storage::register(env);
    crypto::register(env);
    events::register(env);
    math::register(env);
}
