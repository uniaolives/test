// Gerado por ASI-Ω Universal COBOL Parser v2.0
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use rust_decimal::prelude::ToPrimitive;
use std::collections::HashMap;
use thiserror::Error;
use tokio::sync::RwLock;

#[derive(Error, Debug)]
pub enum BusinessError {
    #[error("Variável não encontrada: {0}")]
    VariableNotFound(String),
    #[error("Erro aritmético: {0}")]
    ArithmeticError(String),
}

pub struct Context {
    pub variables: RwLock<HashMap<String, Variable>>,
    pub alter_state: RwLock<HashMap<String, String>>,
}

#[derive(Clone, Debug)]
pub enum Variable {
    Decimal(Decimal, u32, u32),
    String(String, usize),
    Integer(i64),
}

impl Context {
    pub fn new() -> Self {
        let mut alter_state = HashMap::new();
        alter_state.insert("PRIMEIRO-PROCESSAMENTO".to_string(), "SEGUNDO-PROCESSO".to_string());
        Self {
            variables: RwLock::new(HashMap::new()),
            alter_state: RwLock::new(alter_state),
        }
    }

    pub async fn get_alter_target(&self, para: &str) -> String {
        let state = self.alter_state.read().await;
        state.get(para).cloned().unwrap_or_else(|| para.to_string())
    }

    pub async fn set_alter_target(&self, para: &str, new: &str) {
        let mut state = self.alter_state.write().await;
        state.insert(para.to_string(), new.to_string());
    }

    pub async fn get_decimal(&self, name: &str) -> Result<Decimal, BusinessError> {
        let vars = self.variables.read().await;
        match vars.get(name) {
            Some(Variable::Decimal(v, _, _)) => Ok(*v),
            _ => Err(BusinessError::VariableNotFound(name.to_string())),
        }
    }

    pub async fn set_decimal(&self, name: &str, value: Decimal) -> Result<(), BusinessError> {
        let mut vars = self.variables.write().await;
        vars.insert(name.to_string(), Variable::Decimal(value, 9, 2));
        Ok(())
    }

    pub async fn get_string(&self, name: &str) -> Result<String, BusinessError> {
        let vars = self.variables.read().await;
        match vars.get(name) {
            Some(Variable::String(v, _)) => Ok(v.clone()),
            _ => Err(BusinessError::VariableNotFound(name.to_string())),
        }
    }

    pub async fn set_string(&self, name: &str, value: String) -> Result<(), BusinessError> {
        let mut vars = self.variables.write().await;
        vars.insert(name.to_string(), Variable::String(value, 255));
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ParagraphState {
    MAIN_LOGIC,
    INICIALIZAR,
    PRIMEIRO_PROCESSAMENTO,
    VERIFICAR_FLAG,
    SEGUNDO_PROCESSO,
    FIM_PROGRAMA,
    FINISHED,
}

pub async fn main_loop(ctx: &Context) -> Result<(), BusinessError> {
    let mut state = ParagraphState::MAIN_LOGIC;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 10000;

    loop {
        if state == ParagraphState::FINISHED { break; }
        iterations += 1;
        if iterations > MAX_ITERATIONS {
            return Err(BusinessError::ArithmeticError("Loop infinito detectado".to_string()));
        }

        match state {
            ParagraphState::MAIN_LOGIC => {
                Box::pin(inicializar(ctx)).await?;
                state = ParagraphState::PRIMEIRO_PROCESSAMENTO; continue;
            }
            ParagraphState::INICIALIZAR => {
                ctx.set_decimal("WS-COUNTER", dec!(100)).await?;
                ctx.set_string("WS-FLAG", 'N'.to_string()).await?;
                state = ParagraphState::PRIMEIRO_PROCESSAMENTO; continue;
            }
            ParagraphState::PRIMEIRO_PROCESSAMENTO => {
                let actual = ctx.get_alter_target("PRIMEIRO-PROCESSAMENTO").await;
                if actual != "PRIMEIRO-PROCESSAMENTO" {
                    state = match actual.as_str() {
                        "MAIN-LOGIC" => ParagraphState::MAIN_LOGIC,
                        "INICIALIZAR" => ParagraphState::INICIALIZAR,
                        "PRIMEIRO-PROCESSAMENTO" => ParagraphState::PRIMEIRO_PROCESSAMENTO,
                        "VERIFICAR-FLAG" => ParagraphState::VERIFICAR_FLAG,
                        "SEGUNDO-PROCESSO" => ParagraphState::SEGUNDO_PROCESSO,
                        "FIM-PROGRAMA" => ParagraphState::FIM_PROGRAMA,
                        _ => ParagraphState::FINISHED,
                    };
                    continue;
                }
                println!("{}", ctx.get_decimal("WS-COUNTER").await?);
                ctx.set_decimal("WS-COUNTER", ctx.get_decimal("WS-COUNTER").await? - dec!(1)).await?;
                if ctx.get_decimal("WS-COUNTER").await? > dec!(0) {
                state = ParagraphState::PRIMEIRO_PROCESSAMENTO; continue;
                } else {
                state = ParagraphState::VERIFICAR_FLAG; continue;
                }
            }
            ParagraphState::VERIFICAR_FLAG => {
                if ctx.get_string("WS-FLAG").await? == "S" {
                state = ParagraphState::FIM_PROGRAMA; continue;
                } else {
                ctx.set_alter_target("PRIMEIRO-PROCESSAMENTO", "SEGUNDO-PROCESSO").await;
                ctx.set_string("WS-FLAG", 'S'.to_string()).await?;
                state = ParagraphState::PRIMEIRO_PROCESSAMENTO; continue;
                }
            }
            ParagraphState::SEGUNDO_PROCESSO => {
                println!("{}", ctx.get_decimal("WS-COUNTER").await?);
                state = ParagraphState::FIM_PROGRAMA; continue;
            }
            ParagraphState::FIM_PROGRAMA => {
                println!("{}", "FIM");
                state = ParagraphState::FINISHED; continue;
            }
            ParagraphState::FINISHED => break,
        }
    }
    Ok(())
}
async fn main_logic(ctx: &Context) -> Result<(), BusinessError> {
    Box::pin(inicializar(ctx)).await?;
    // Jump GO_TO ignored in structured version
    Ok(())
}
async fn inicializar(ctx: &Context) -> Result<(), BusinessError> {
    ctx.set_decimal("WS-COUNTER", dec!(100)).await?;
    ctx.set_string("WS-FLAG", 'N'.to_string()).await?;
    Ok(())
}
async fn primeiro_processamento(ctx: &Context) -> Result<(), BusinessError> {
    println!("{}", ctx.get_decimal("WS-COUNTER").await?);
    ctx.set_decimal("WS-COUNTER", ctx.get_decimal("WS-COUNTER").await? - dec!(1)).await?;
    if ctx.get_decimal("WS-COUNTER").await? > dec!(0) {
    // Jump GO_TO ignored in structured version
    } else {
    // Jump GO_TO ignored in structured version
    }
    Ok(())
}
async fn verificar_flag(ctx: &Context) -> Result<(), BusinessError> {
    if ctx.get_string("WS-FLAG").await? == "S" {
    // Jump GO_TO ignored in structured version
    } else {
    // TODO: ALTER
    ctx.set_string("WS-FLAG", 'S'.to_string()).await?;
    // Jump GO_TO ignored in structured version
    }
    Ok(())
}
async fn segundo_processo(ctx: &Context) -> Result<(), BusinessError> {
    println!("{}", ctx.get_decimal("WS-COUNTER").await?);
    // Jump GO_TO ignored in structured version
    Ok(())
}
async fn fim_programa(ctx: &Context) -> Result<(), BusinessError> {
    println!("{}", "FIM");
    // Jump STOP_RUN ignored in structured version
    Ok(())
}