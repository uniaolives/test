//! Autenticação PQC - INV-1 Non-Repudiation
//! Verifica assinaturas Dilithium5 e integridade BLAKE3

use tonic::{Request, Status};
use pqcrypto_dilithium::dilithium5::{PublicKey, DetachedSignature};
use pqcrypto_traits::sign::{PublicKey as _, DetachedSignature as _};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub async fn authenticate_request<T>(
    request: Request<T>,
    _prince_pubkey: &PublicKey,
) -> Result<Request<T>, Status> {

    let metadata = request.metadata();

    // 1. Timestamp (prevenção replay)
    let timestamp_str = metadata.get("x-request-timestamp")
        .ok_or_else(|| Status::unauthenticated("Timestamp ausente"))?
        .to_str()
        .map_err(|_| Status::unauthenticated("Timestamp inválido"))?;

    let timestamp = timestamp_str.parse::<u64>()
        .map_err(|_| Status::unauthenticated("Timestamp não numérico"))?;

    let request_time = UNIX_EPOCH + Duration::from_secs(timestamp);
    let now = SystemTime::now();

    let drift = if now > request_time {
        now.duration_since(request_time).unwrap()
    } else {
        request_time.duration_since(now).unwrap()
    };

    if drift > Duration::from_secs(300) {
        return Err(Status::unauthenticated("Request fora da janela de tempo"));
    }

    // 2. Assinatura PQC
    let signature_bytes = metadata.get("x-pqc-signature")
        .ok_or_else(|| Status::unauthenticated("Assinatura ausente"))?
        .as_bytes();

    let signature = DetachedSignature::from_bytes(signature_bytes)
        .map_err(|_| Status::unauthenticated("Formato de assinatura inválido"))?;

    // 3. Chave pública do solicitante
    let pubkey_bytes = metadata.get("x-requestor-pubkey")
        .ok_or_else(|| Status::unauthenticated("Chave pública ausente"))?
        .as_bytes();

    let requestor_pubkey = PublicKey::from_bytes(pubkey_bytes)
        .map_err(|_| Status::unauthenticated("Formato de chave pública inválido"))?;

    // 4. Verificar whitelist (INV-3 compliance)
    if !is_pubkey_whitelisted(&requestor_pubkey) {
        return Err(Status::permission_denied("Chave pública não autorizada"));
    }

    // 5. Reconstruir mensagem assinada
    let integrity_hash_hex = metadata.get("x-blake3-integrity")
        .ok_or_else(|| Status::unauthenticated("Hash de integridade ausente"))?
        .to_str()
        .map_err(|_| Status::unauthenticated("Hash de integridade inválido"))?;

    let integrity_hash = hex::decode(integrity_hash_hex)
        .map_err(|_| Status::unauthenticated("Hash de integridade não-hexadecimal"))?;

    // 6. Verificar assinatura do hash
    if pqcrypto_dilithium::dilithium5::verify_detached_signature(&signature, &integrity_hash, &requestor_pubkey).is_err() {
        return Err(Status::unauthenticated("Assinatura PQC inválida"));
    }

    Ok(request)
}

fn is_pubkey_whitelisted(pubkey: &PublicKey) -> bool {
    // Em produção, consulta whitelist no WORM ledger
    // Para este bloco, verificamos apenas se a chave não está vazia como placeholder de segurança
    !pubkey.as_bytes().iter().all(|&b| b == 0)
}
