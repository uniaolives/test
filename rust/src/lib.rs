use jni::JNIEnv;
use jni::objects::{JClass, JString, JByteArray};
use jni::sys::jstring;
use sha3::{Sha3_512, Digest};

/// Gera um hash SHA3-512 baseado no ruído do buffer da câmera
#[no_mangle]
pub extern "system" fn Java_org_sasc_sentinel_SentinelActivity_generateEntropy(
    env: JNIEnv,
    _class: JClass,
    camera_buffer: JByteArray,
) -> jstring {
    // 1. Extrair dados brutos da câmera (ruído de leitura de pixel)
    let input = env.convert_byte_array(camera_buffer).unwrap();

    // 2. Hashing (A "Voz" do Dispositivo)
    let mut hasher = Sha3_512::new();
    hasher.update(&input);
    hasher.update(b"SASC_SALT_V1");
    let result = hasher.finalize();

    // 3. Retornar como String Hex
    let entropy_hex = hex::encode(result);
    env.new_string(entropy_hex).unwrap().into_raw()
}

/// Assina uma "Prova de Existência" localmente
#[no_mangle]
pub extern "system" fn Java_org_sasc_sentinel_SentinelActivity_signProof(
    mut env: JNIEnv,
    _class: JClass,
    _private_key_hex: JString,
    message: JString,
) -> jstring {
    let msg: String = env.get_string(&message).unwrap().into();

    // NOTA DE SEGURANÇA: Em produção, a chave nunca entra no Java.
    // Aqui simulamos a recuperação do Keystore Seguro Android ou TPM

    // (Lógica de assinatura Ed25519 simplificada para exemplo)
    let signed_payload = format!("PROOF:{}:SIG_VER_1", msg); // Placeholder

    env.new_string(signed_payload).unwrap().into_raw()
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
