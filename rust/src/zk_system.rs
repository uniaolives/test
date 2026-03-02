// zk_system.rs
// Sistema completo de geração e verificação de provas constitucionais

use bellman::groth16::{
    create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof,
    Parameters, Proof, PreparedVerifyingKey,
};
use rand::rngs::OsRng;
use bls12_381::{Bls12, Scalar};
use crate::zk_vajra_circuit::ConstitutionalCircuit;

pub struct ZKCruxSystem {
    pub params: Parameters<Bls12>,
    pub pvk: PreparedVerifyingKey<Bls12>,
}

impl ZKCruxSystem {
    pub fn setup() -> Self {
        // Cerimônia de setup confiável para o circuito constitucional
        let circuit = ConstitutionalCircuit {
            input_vector: vec![Scalar::from(0); 1024],
            neural_weights: vec![Scalar::from(0); 100], // Reduzido para o demo
            energy_consumed: Scalar::from(0),
            hdc_score: Scalar::from(0),
            output_hash: Scalar::from(0),
            max_energy_limit: Scalar::from(1000),
            min_hdc_threshold: Scalar::from(720),
            merkle_root: Scalar::from(0),
        };

        let params = generate_random_parameters::<Bls12, _, _>(circuit, &mut OsRng).unwrap();
        let pvk = prepare_verifying_key(&params.vk);

        ZKCruxSystem { params, pvk }
    }

    pub fn prove_inference(
        &self,
        input: &[Scalar],
        weights: &[Scalar],
        energy: f64,
        hdc: f64,
        output_hash: Scalar,
        merkle_root: Scalar,
    ) -> Result<Proof<Bls12>, String> {

        let circuit = ConstitutionalCircuit {
            input_vector: input.to_vec(),
            neural_weights: weights.to_vec(),
            energy_consumed: Scalar::from((energy * 1000.0) as u64),
            hdc_score: Scalar::from((hdc * 1000.0) as u64),
            output_hash,
            max_energy_limit: Scalar::from(1000),
            min_hdc_threshold: Scalar::from(720),
            merkle_root,
        };

        let proof = create_random_proof(circuit, &self.params, &mut OsRng)
            .map_err(|e| format!("Falha na geração da prova: {}", e))?;

        Ok(proof)
    }

    pub fn verify_constitutionality(
        &self,
        proof: &Proof<Bls12>,
        public_inputs: &[Scalar],
    ) -> Result<bool, String> {

        match verify_proof(&self.pvk, proof, public_inputs) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}
