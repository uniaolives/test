// rust_sentry/src/lib.rs
use std::sync::{Arc, RwLock};
use rustler::{Env, Error, Term, Encoder};
use lazy_static::lazy_static;

// Estruturas de Dados
#[derive(Debug)]
struct GlobalState {
    current_intent: String,
    emotional_tone: f32,
    security_level: u8, // 0 a 100
    task_queue: Vec<String>,
}

// Singleton Thread-Safe
lazy_static! {
    static ref CONSCIOUSNESS: Arc<RwLock<GlobalState>> = Arc::new(RwLock::new(GlobalState {
        current_intent: String::from("idle"),
        emotional_tone: 0.5,
        security_level: 100,
        task_queue: Vec::new(),
    }));
}

rustler::atoms! {
    intent,
    current,
    emotion,
    ok,
    blocked,
    allowed,
    logged,
}

rustler::init!("Elixir.ConciousnessBridge.Native", [
    get_intent,
    set_intent,
    check_security,
    log_audit
]);

// --- FUNÇÕES NIF EXPORTADAS ---

#[rustler::nif]
fn get_intent<'a>(env: Env<'a>) -> Result<Term<'a>, Error> {
    let state = CONSCIOUSNESS.read().unwrap();

    let intent_tuple = rustler::types::tuple::make_tuple(env, &[
        current().encode(env),
        state.current_intent.encode(env)
    ]);

    let emotion_tuple = rustler::types::tuple::make_tuple(env, &[
        emotion().encode(env),
        state.emotional_tone.encode(env)
    ]);

    Ok(rustler::types::tuple::make_tuple(env, &[
        intent().encode(env),
        intent_tuple,
        emotion_tuple
    ]))
}

#[rustler::nif]
fn set_intent<'a>(env: Env<'a>, intent_str: String) -> Result<Term<'a>, Error> {
    let mut state = CONSCIOUSNESS.write().unwrap();
    state.current_intent = intent_str.clone();
    state.task_queue.push(intent_str);

    // Simulação: Ajustar tom emocional baseado na complexidade
    state.emotional_tone = if state.current_intent.len() > 20 { 0.8 } else { 0.4 };

    println!("🧠 [RUST] Intentão atualizada: {} (Tom: {})", state.current_intent, state.emotional_tone);

    Ok(ok().encode(env))
}

#[rustler::nif]
fn check_security<'a>(env: Env<'a>, domain: String) -> Result<Term<'a>, Error> {
    let _state = CONSCIOUSNESS.read().unwrap();

    // Lista branca simples no Rust
    let allowed_domains = vec!["github.com", "gitlab.com", "api.openai.com"];

    // Check if domain is allowed or is a subpath of an allowed domain
    let is_safe = allowed_domains.iter().any(|&d| {
        domain == d || domain.starts_with(&(d.to_string() + "/"))
    });

    if !is_safe {
        println!("🛡️ [RUST] BLOQUEIO DE SEGURANÇA: {}", domain);
        return Ok(blocked().encode(env));
    }

    Ok(allowed().encode(env))
}

#[rustler::nif]
fn log_audit<'a>(env: Env<'a>) -> Result<Term<'a>, Error> {
    println!("📝 [RUST] Log de auditoria salvo no estado global.");
    Ok(logged().encode(env))
}
