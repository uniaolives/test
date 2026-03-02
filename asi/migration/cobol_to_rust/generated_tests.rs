#[cfg(test)]
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
        ctx.set_decimal("WS-PRINCIPAL", dec!(100.00)).await.unwrap();
        ctx.set_decimal("WS-PERIODS", dec!(100.00)).await.unwrap();
        inicializar(&mut ctx).await.unwrap();
    }

    #[tokio::test]
    async fn test_inicializar_1() {
        let mut ctx = Context::new();
        ctx.set_decimal("WS-RATE", dec!(999999999.99)).await.unwrap();
        ctx.set_decimal("WS-PRINCIPAL", dec!(999999999.99)).await.unwrap();
        ctx.set_decimal("WS-PERIODS", dec!(999999999.99)).await.unwrap();
        inicializar(&mut ctx).await.unwrap();
    }

    #[tokio::test]
    async fn test_calcular_juros_0() {
        let mut ctx = Context::new();
        ctx.set_decimal("WS-INTEREST", dec!(100.00)).await.unwrap();
        ctx.set_decimal("WS-RATE", dec!(100.00)).await.unwrap();
        ctx.set_decimal("WS-PRINCIPAL", dec!(100.00)).await.unwrap();
        ctx.set_decimal("WS-CATEGORY", dec!(100.00)).await.unwrap();
        ctx.set_decimal("WS-PERIODS", dec!(100.00)).await.unwrap();
        calcular_juros(&mut ctx).await.unwrap();
    }

    #[tokio::test]
    async fn test_calcular_juros_1() {
        let mut ctx = Context::new();
        ctx.set_decimal("WS-INTEREST", dec!(999999999.99)).await.unwrap();
        ctx.set_decimal("WS-RATE", dec!(999999999.99)).await.unwrap();
        ctx.set_decimal("WS-PRINCIPAL", dec!(999999999.99)).await.unwrap();
        ctx.set_decimal("WS-CATEGORY", dec!(999999999.99)).await.unwrap();
        ctx.set_decimal("WS-PERIODS", dec!(999999999.99)).await.unwrap();
        calcular_juros(&mut ctx).await.unwrap();
    }

    #[tokio::test]
    async fn test_calcular_juros_2() {
        let mut ctx = Context::new();
        ctx.set_decimal("WS-INTEREST", dec!(0.00)).await.unwrap();
        ctx.set_decimal("WS-RATE", dec!(0.00)).await.unwrap();
        ctx.set_decimal("WS-PRINCIPAL", dec!(0.00)).await.unwrap();
        ctx.set_decimal("WS-CATEGORY", dec!(0.00)).await.unwrap();
        ctx.set_decimal("WS-PERIODS", dec!(0.00)).await.unwrap();
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