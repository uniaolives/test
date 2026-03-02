//! Cliente SASC-SOCIETY v30.15-Ω - Seguro e Certificado

use sasc_society::hsm_signer::HsmManager;
use sasc_society::vajra_sasc_bridge::{assess_with_vajra, SecuredMetadata};
use sasc_society::gaia_integration::GaiaNetManager;
use sasc_society::constants::*;
use sasc_society::grpc::sasc_society_proto::{
    sot_orchestrator_client::SotOrchestratorClient,
    ProcessDecisionRequest,
};
use tonic::transport::Channel;
use tonic::Request;
use std::sync::Arc;
use pqcrypto_traits::sign::PublicKey as _;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SASC Client Secure");
    Ok(())
}

pub struct SascSecureClient {
    hsm: Arc<HsmManager>,
    gaia: Arc<GaiaNetManager>,
    channel: Channel,
}

impl SascSecureClient {
    pub async fn new(gaia_sensor_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            hsm: Arc::new(HsmManager::new()?),
            gaia: Arc::new(GaiaNetManager::new(gaia_sensor_path)?),
            channel: Channel::from_static("http://[::1]:50051").connect().await?,
        })
    }
}
