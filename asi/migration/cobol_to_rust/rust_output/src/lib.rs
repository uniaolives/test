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
    #[error("Divisão por zero")]
    DivisionByZero,
}

/// Contexto de execução (equivalente à WORKING-STORAGE)
pub struct Context {
    pub variables: RwLock<HashMap<String, Variable>>,
}

#[derive(Clone, Debug)]
pub enum Variable {
    Decimal(Decimal, u32, u32), // valor, int_digits, dec_digits
    String(String, usize),      // valor, tamanho máximo
    Integer(i64),
}

impl Context {
    pub fn new() -> Self {
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