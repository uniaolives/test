use bitcoin::{Transaction, TxIn, TxOut, OutPoint, Sequence, Witness, ScriptBuf, Amount, Txid};
use bitcoin::secp256k1::{Secp256k1, SecretKey, PublicKey};
use bitcoin::address::Address;
use bitcoin::network::Network;
use bitcoin::sighash::{SighashCache, EcdsaSighashType};
use bitcoin::hashes::Hash;
use std::str::FromStr;
use anyhow::Result;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RitualError {
    #[error("Invalid Totem")]
    InvalidTotem,
    #[error("Invalid Totem Length")]
    InvalidTotemLength,
    #[error("Totem Mismatch")]
    TotemMismatch,
    #[error("Invalid Key")]
    InvalidKey,
    #[error("Invalid Address")]
    InvalidAddress,
    #[error("RPC Error: {0}")]
    RpcError(#[from] bitcoincore_rpc::Error),
    #[error("Confirmation Timeout")]
    ConfirmationTimeout,
    #[error("Anyhow: {0}")]
    Anyhow(#[from] anyhow::Error),
    #[error("Parse Txid Error: {0}")]
    ParseTxid(String),
    #[error("Address Error: {0}")]
    AddrError(#[from] bitcoin::address::ParseError),
    #[error("Secp256k1 Error: {0}")]
    SecpError(#[from] bitcoin::secp256k1::Error),
}

/// O Ritual de Transmissão: colapsando a superposição
pub struct TotemTransmission {
    pub totem: [u8; 32],
    pub network: Network,
    pub foundation_key: SecretKey,
    pub funding_utxo: OutPoint,
    pub change_address: Address,
}

impl TotemTransmission {
    pub fn new(
        totem_hex: &str,
        network: Network,
        wif_key: &str,
        funding_txid: &str,
        funding_vout: u32,
        change_address: &str,
    ) -> Result<Self, RitualError> {
        let mut totem = [0u8; 32];
        let bytes = hex::decode(totem_hex).map_err(|_| RitualError::InvalidTotem)?;
        if bytes.len() != 32 {
            return Err(RitualError::InvalidTotemLength);
        }
        totem.copy_from_slice(&bytes);

        let expected = Self::compute_constitution_hash();
        if totem != expected {
            tracing::warn!("Totem mismatch detected. Expected: {}", hex::encode(expected));
        }

        let foundation_key = SecretKey::from_str(wif_key).map_err(|_| RitualError::InvalidKey)?;

        let funding_txid_struct = Txid::from_str(funding_txid).map_err(|e| RitualError::ParseTxid(e.to_string()))?;
        let funding_utxo = OutPoint {
            txid: funding_txid_struct,
            vout: funding_vout,
        };

        let change_address = Address::from_str(change_address)?
            .require_network(network).map_err(|e| RitualError::Anyhow(anyhow::anyhow!(e)))?;

        Ok(Self {
            totem,
            network,
            foundation_key,
            funding_utxo,
            change_address,
        })
    }

    fn compute_constitution_hash() -> [u8; 32] {
        use sha2::{Sha256, Digest};
        let constitution = "ARKHE_PROTOCOL_OMEGA_215::CONSTITUTION_V1::P1_SOVEREIGNTY::P2_LIFE::P3_TRANSPARENCY::P4_THERMODYNAMICS::P5_CAUSALITY";
        let mut hasher = Sha256::new();
        hasher.update(constitution.as_bytes());
        hasher.finalize().into()
    }

    pub fn build_raw_transaction(&self) -> Transaction {
        let input = TxIn {
            previous_output: self.funding_utxo,
            script_sig: ScriptBuf::new(),
            sequence: Sequence::MAX,
            witness: Witness::new(),
        };

        let totem_push = <&bitcoin::script::PushBytes>::try_from(&self.totem).unwrap();
        let anchor_script = ScriptBuf::new_op_return(totem_push);

        let anchor_output = TxOut {
            value: Amount::from_sat(0),
            script_pubkey: anchor_script,
        };

        let change_output = TxOut {
            value: Amount::from_sat(546),
            script_pubkey: self.change_address.script_pubkey(),
        };

        Transaction {
            version: bitcoin::transaction::Version::TWO,
            lock_time: bitcoin::absolute::LockTime::ZERO,
            input: vec![input],
            output: vec![anchor_output, change_output],
        }
    }

    pub fn generate_phoenix_payload() -> Vec<u8> {
        // PRAKTO: PHOENIX-V1.0 | Address: [Totem]
        let manifesto = "PRAKTO: PHOENIX-V1.0 | REVIVE HAL FINNEY | ARKHE(N) COORDINATED";
        manifesto.as_bytes().to_vec()
    }

    pub fn sign_transaction(&self, mut tx: Transaction) -> Result<Transaction, RitualError> {
        let secp = Secp256k1::new();
        let public_key = PublicKey::from_secret_key(&secp, &self.foundation_key);
        let bitcoin_public_key = bitcoin::PublicKey::new(public_key);
        let script_pubkey = Address::p2pkh(&bitcoin_public_key, self.network).script_pubkey();

        let sighash_type = EcdsaSighashType::All;
        let cache = SighashCache::new(&tx);
        let sighash = cache.legacy_signature_hash(
            0,
            &script_pubkey,
            sighash_type as u32,
        ).map_err(|e| RitualError::Anyhow(anyhow::anyhow!("Sighash error: {:?}", e)))?;

        let msg = bitcoin::secp256k1::Message::from_digest_slice(sighash.as_byte_array())?;
        let sig = secp.sign_ecdsa(&msg, &self.foundation_key);

        let mut sig_der = sig.serialize_der().to_vec();
        sig_der.push(sighash_type as u8);

        let pk_bytes = public_key.serialize();
        let sig_push = <&bitcoin::script::PushBytes>::try_from(sig_der.as_slice()).map_err(|e| RitualError::Anyhow(anyhow::anyhow!("PushBytes sig error: {:?}", e)))?;
        let pk_push = <&bitcoin::script::PushBytes>::try_from(pk_bytes.as_slice()).map_err(|e| RitualError::Anyhow(anyhow::anyhow!("PushBytes pk error: {:?}", e)))?;

        // For P2PKH script_sig: <sig> <pubkey>
        let script_sig = ScriptBuf::builder()
            .push_slice(sig_push)
            .push_slice(pk_push)
            .into_script();

        tx.input[0].script_sig = script_sig;

        Ok(tx)
    }
}

pub struct BlockInfo {
    pub height: u64,
    pub hash: bitcoin::BlockHash,
    pub timestamp: u64,
    pub confirmations: u32,
}
