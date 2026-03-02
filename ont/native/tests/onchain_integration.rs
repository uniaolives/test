#[cfg(test)]
mod integration_tests {
    use ontology_lang::parse_program;
    use ontology_lang::onchain::{
        OnChainAutomation,
        BlockchainTarget,
        VerificationLevel,
        OnChainError
    };
    use std::fs;
    use tokio;

    #[tokio::test]
    async fn test_sasc_tmr_deployment() {
        // Carregar programa DAO de exemplo
        let source = fs::read_to_string("examples/onchain/dao.onto")
            .expect("Failed to read DAO example");

        let program = parse_program(&source)
            .expect("Failed to parse DAO program");

        // Criar automação com TMR
        let automation = OnChainAutomation::new(
            BlockchainTarget::SASC,
            VerificationLevel::TMR,
            true
        );

        // Testar compilação
        let compiled = automation.compile_to_smart_contract(&program)
            .expect("Compilation failed");

        assert!(compiled.source_code.contains("function vote"));
        assert!(compiled.stats.functions_compiled > 0);
        assert!(compiled.stats.transmutations_applied > 0);

        println!("✅ Compilação SASC bem-sucedida:");
        println!("   Funções: {}", compiled.stats.functions_compiled);
        println!("   Transmutações: {}", compiled.stats.transmutations_applied);
        println!("   Constraints: {}", compiled.stats.diplomatic_constraints);
    }

    #[tokio::test]
    async fn test_paradigm_constraint_violation() {
        // Programa que viola constraints
        let source = r#"
            fn bad_function : Pure<Int> -> Pure<Int> =
                {{ state_modifying_operation }}

            transmute bad_function to Imperative with
                sandbox(StateSandbox),
                O(1) time
        "#;

        let program = parse_program(source)
            .expect("Failed to parse program");

        let automation = OnChainAutomation::new(
            BlockchainTarget::SASC,
            VerificationLevel::FullSASC,
            true
        );

        // Deve falhar na verificação de constraints
        match automation.automate(&program).await {
            Err(OnChainError::ConstraintViolation(_)) => {
                println!("✅ Violação de constraint detectada corretamente");
            }
            Err(e) => panic!("Erro inesperado: {:?}", e),
            Ok(_) => panic!("Deveria ter falhado por violação de constraint"),
        }
    }

    #[tokio::test]
    async fn test_bytecode_integrity() {
        use ontology_lang::onchain::deployer::sasc;

        // Carregar programa
        let source = fs::read_to_string("examples/onchain/oracle.onto")
            .expect("Failed to read oracle example");

        let program = parse_program(&source)
            .expect("Failed to parse oracle program");

        // Compilar
        let automation = OnChainAutomation::new(
            BlockchainTarget::SASC,
            VerificationLevel::TMR,
            true
        );

        let compiled = automation.compile_to_smart_contract(&program)
            .expect("Compilation failed");

        // Verificar que o bytecode pode ser gerado
        assert!(compiled.source_code.len() > 100);

        // Verificar constraints
        let constraints = sasc::extract_constraints(&compiled);
        assert!(constraints.iter().any(|c| c.contains("sandbox")),
                "Deve conter constraint sandbox");

        println!("✅ Verificação de integridade de bytecode passou");
        println!("   Constraints extraídas: {:?}", constraints);
    }
}
