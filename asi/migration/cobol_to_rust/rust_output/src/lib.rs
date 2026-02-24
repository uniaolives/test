// Gerado por ASI-Ω Universal COBOL Parser v2.0
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use rust_decimal::prelude::ToPrimitive;
use std::collections::HashMap;
use thiserror::Error;
use tokio::sync::RwLock;
// Gerado por ASI-Ω COBOL Transmuter
// Fonte: PROCEDURE DIVISION
// Parágrafos: 4
// Entry point: MAIN-LOGIC

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use thiserror::Error;
use tokio::sync::RwLock;
use std::pin::Pin;
use std::future::Future;

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
    #[error("Divisão por zero")]
    DivisionByZero,
}

/// Contexto de execução (equivalente à WORKING-STORAGE)
pub struct Context {
    pub variables: RwLock<HashMap<String, Variable>>,
}

#[derive(Clone, Debug)]
pub enum Variable {
    Decimal(Decimal, u32, u32),
    String(String, usize),
    Decimal(Decimal, u32, u32), // valor, int_digits, dec_digits
    String(String, usize),      // valor, tamanho máximo
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

        Self {
            variables: RwLock::new(HashMap::new()),
        }
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
        // Preservar metadados de PIC se existirem
        let (int_d, dec_d) = if let Some(Variable::Decimal(_, i, d)) = vars.get(name) {
            (*i, *d)
        } else {
            (9, 2) // default
        };
        vars.insert(name.to_string(), Variable::Decimal(value, int_d, dec_d));
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
        let max_len = if let Some(Variable::String(_, m)) = vars.get(name) {
            *m
        } else {
            255
        };
        let truncated = if value.len() > max_len {
            value[..max_len].to_string()
        } else {
            value
        };
        vars.insert(name.to_string(), Variable::String(truncated, max_len));
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
/// Parágrafo COBOL: MAIN-LOGIC
/// Linhas: 5-8
/// Dependências de leitura: nenhuma
/// Dependências de escrita: nenhuma
pub async fn main_logic(ctx: &mut Context) -> Result<(), BusinessError> {
    Box::pin(inicializar(ctx)).await?;
    Box::pin(calcular_juros(ctx)).await?;
    Box::pin(finalizar(ctx)).await?;
    return Ok(());
}

/// Parágrafo COBOL: CALCULAR-JUROS
/// Linhas: 16-20
/// Dependências de leitura: WS-PRINCIPAL, WS-RATE, WS-PERIODS, WS-INTEREST, WS-CATEGORY
/// Dependências de escrita: 1, WS-INTEREST, 0
pub async fn calcular_juros(ctx: &mut Context) -> Result<(), BusinessError> {
    ctx.set_decimal("WS-INTEREST", (ctx.get_decimal("WS-PRINCIPAL").await? * ctx.get_decimal("WS-RATE").await? / dec!(100) * ctx.get_decimal("WS-PERIODS").await?).round_dp(2)).await?;
    if ctx.get_decimal("WS-INTEREST").await? > dec!(1000) {
    ctx.set_decimal("WS-CATEGORY", dec!(1)).await?;
    } else {
    ctx.set_decimal("WS-CATEGORY", dec!(0)).await?;
    }
    Ok(())
}

/// Parágrafo COBOL: INICIALIZAR
/// Linhas: 11-13
/// Dependências de leitura: WS-RATE, WS-PERIODS, WS-PRINCIPAL
/// Dependências de escrita: 5.25, 10000, 12
pub async fn inicializar(ctx: &mut Context) -> Result<(), BusinessError> {
    ctx.set_decimal("WS-PRINCIPAL", dec!(10000)).await?;
    ctx.set_decimal("WS-RATE", dec!(5.25)).await?;
    ctx.set_decimal("WS-PERIODS", dec!(12)).await?;
    Ok(())
}

/// Parágrafo COBOL: FINALIZAR
/// Linhas: 23-23
/// Dependências de leitura: nenhuma
/// Dependências de escrita: nenhuma
pub async fn finalizar(ctx: &mut Context) -> Result<(), BusinessError> {
    println!("{}", "FIM");
    Ok(())
}

/// Ponto de entrada do programa transmutado
#[tokio::main]
pub async fn main_entry() -> Result<(), BusinessError> {
    let mut ctx = Context::new();
    main_logic(&mut ctx).await
}#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_main_logic_0() {
        let mut ctx = Context::new();
        main_logic(&mut ctx).await.unwrap();
    }

    #[tokio::test]
    async fn test_main_logic_1() {
        let mut ctx = Context::new();
        main_logic(&mut ctx).await.unwrap();
    }

    #[tokio::test]
    async fn test_inicializar_0() {
        let mut ctx = Context::new();
        ctx.set_decimal("WS-RATE", dec!(100.00)).await.unwrap();
        ctx.set_decimal("WS-PERIODS", dec!(100.00)).await.unwrap();
        ctx.set_decimal("WS-PRINCIPAL", dec!(100.00)).await.unwrap();
        inicializar(&mut ctx).await.unwrap();
    }

    #[tokio::test]
    async fn test_inicializar_1() {
        let mut ctx = Context::new();
        ctx.set_decimal("WS-RATE", dec!(999999999.99)).await.unwrap();
        ctx.set_decimal("WS-PERIODS", dec!(999999999.99)).await.unwrap();
        ctx.set_decimal("WS-PRINCIPAL", dec!(999999999.99)).await.unwrap();
        inicializar(&mut ctx).await.unwrap();
    }

    #[tokio::test]
    async fn test_calcular_juros_0() {
        let mut ctx = Context::new();
        ctx.set_decimal("WS-PRINCIPAL", dec!(100.00)).await.unwrap();
        ctx.set_decimal("WS-RATE", dec!(100.00)).await.unwrap();
        ctx.set_decimal("WS-PERIODS", dec!(100.00)).await.unwrap();
        ctx.set_decimal("WS-INTEREST", dec!(100.00)).await.unwrap();
        ctx.set_decimal("WS-CATEGORY", dec!(100.00)).await.unwrap();
        calcular_juros(&mut ctx).await.unwrap();
    }

    #[tokio::test]
    async fn test_calcular_juros_1() {
        let mut ctx = Context::new();
        ctx.set_decimal("WS-PRINCIPAL", dec!(999999999.99)).await.unwrap();
        ctx.set_decimal("WS-RATE", dec!(999999999.99)).await.unwrap();
        ctx.set_decimal("WS-PERIODS", dec!(999999999.99)).await.unwrap();
        ctx.set_decimal("WS-INTEREST", dec!(999999999.99)).await.unwrap();
        ctx.set_decimal("WS-CATEGORY", dec!(999999999.99)).await.unwrap();
        calcular_juros(&mut ctx).await.unwrap();
    }

    #[tokio::test]
    async fn test_calcular_juros_2() {
        let mut ctx = Context::new();
        ctx.set_decimal("WS-PRINCIPAL", dec!(0.00)).await.unwrap();
        ctx.set_decimal("WS-RATE", dec!(0.00)).await.unwrap();
        ctx.set_decimal("WS-PERIODS", dec!(0.00)).await.unwrap();
        ctx.set_decimal("WS-INTEREST", dec!(0.00)).await.unwrap();
        ctx.set_decimal("WS-CATEGORY", dec!(0.00)).await.unwrap();
        calcular_juros(&mut ctx).await.unwrap();
    }

    #[tokio::test]
    async fn test_finalizar_0() {
        let mut ctx = Context::new();
        finalizar(&mut ctx).await.unwrap();
    }

    #[tokio::test]
    async fn test_finalizar_1() {
        let mut ctx = Context::new();
        finalizar(&mut ctx).await.unwrap();
    }

}