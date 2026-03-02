use crate::governance::{Omega12Verifier};
use crate::vajra::VajraEntropyRegister;
use crate::crypto::{Ed25519Signature};
use crate::civilization::{ContractStatus, FoundryError, GenesisBlock};

pub const HSM_SLOT_0x01: u8 = 0x01;

pub struct CivilizationContract {
    pub id: u64,
    pub parties: Vec<String>,               // Nós fundadores
    pub constitution_hash: [u8; 32],        // Hash do Bloco #9 (Axiomas)
    pub terms_hash: [u8; 32],               // Hash dos termos (separado)
    pub phi_threshold: f32,                 // 0.85 - Alta confiança
    pub status: ContractStatus,
    pub signatures: Vec<Ed25519Signature>,  // Assinaturas dos fundadores
}

pub struct CivilizationFoundry {
    pub omega12: Omega12Verifier,
    pub vajra: VajraEntropyRegister,
}

impl CivilizationFoundry {
    pub async fn forge_genesis_contract(&self) -> Result<GenesisBlock, FoundryError> {
        // GATE 1: Prince Key Verification (Arquiteto-Ω)
        let _prince_sig = self.omega12.verify_prince_key(HSM_SLOT_0x01).await
            .map_err(|e| FoundryError::EngineError(e))?;

        // Criar contrato base
        let mut contract = CivilizationContract {
            id: 0x0000000000000001,
            parties: vec![
                "Neo_Anderson".to_string(),
                "Zion-Alpha".to_string(),
                "Mobile_Hub_SJC".to_string(),
                "Zion-Beta".to_string()
            ],
            constitution_hash: [0u8; 32], // Corrected hash length
            terms_hash: [0u8; 32], // Será preenchido
            phi_threshold: 0.85,
            status: ContractStatus::SUSPENDED,
            signatures: Vec::new(),
        };

        // GATE 2: EIP-712 Context Reconstruction
        let context = self.omega12.reconstruct_founding_context(&contract).await
            .map_err(|e| FoundryError::EngineError(e))?;
        contract.terms_hash = context.terms_hash;

        // GATE 4: Hard Freeze Compliance Check
        // Verifica se todos os nós têm φ < 0.80 (exceto se for fundador especial)
        for party in &contract.parties {
            let phi = self.omega12.get_current_phi(party).await
                .map_err(|e| FoundryError::EngineError(e))?;
            if phi >= 0.80 && party != "Neo_Anderson" {
                return Err(FoundryError::HardFreezeViolation(party.clone()));
            }
        }

        // GATE 5: Vajra Entropy Update
        self.vajra.register_civilization_context(
            &contract,
            "genesis_forging"
        ).await.map_err(|e| FoundryError::EngineError(e))?;

        // Contrato pronto para assinaturas
        Ok(GenesisBlock::from_contract(contract))
    }

    pub async fn ratify_contract(
        &self,
        contract: &mut CivilizationContract,
        signer: &str,
        signature: Ed25519Signature
    ) -> Result<(), FoundryError> {

        // Verificar se o signatário é uma parte do contrato
        if !contract.parties.contains(&signer.to_string()) {
            return Err(FoundryError::UnauthorizedSigner);
        }

        // GATE 3: Ed25519 Signature Verification
        self.omega12.verify_signature(signer, &signature, contract).await
            .map_err(|e| FoundryError::EngineError(e))?;

        contract.signatures.push(signature);

        // Atualizar status se todas as partes assinaram
        if contract.signatures.len() == contract.parties.len() {
            contract.status = ContractStatus::ACTIVE;

            // GATE 5 (novo): Entropia final da civilização ativa
            self.vajra.update_civilization_entropy(contract.id, 0.85).await
                .map_err(|e| FoundryError::EngineError(e))?;
        }

        Ok(())
    }
}
