use libp2p::{gossipsub, mdns, swarm::NetworkBehaviour};

pub mod multimodal_anchor;
pub mod stack;

#[derive(NetworkBehaviour)]
pub struct ArkheNetBehavior {
    pub gossipsub: gossipsub::Behaviour,
    pub mdns: mdns::tokio::Behaviour,
}
