pub mod protocol;
pub mod node;
pub mod bio;
pub mod bio_socket;
pub mod mqtt;
pub mod multimodal_anchor;
pub mod stack;

pub use protocol::HandoverData;
pub use node::P2PNode;
pub use bio::BioAntenna;
pub use bio_socket::start_bio_server;

use libp2p::{gossipsub, mdns, swarm::NetworkBehaviour};

#[derive(NetworkBehaviour)]
pub struct ArkheNetBehavior {
    pub gossipsub: gossipsub::Behaviour,
    pub mdns: mdns::tokio::Behaviour,
}
