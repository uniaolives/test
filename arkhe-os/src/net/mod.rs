pub mod protocol;
pub mod node;
pub mod bio;
pub mod bio_socket;
use libp2p::{gossipsub, mdns, swarm::NetworkBehaviour};

pub mod multimodal_anchor;
pub mod stack;
pub mod protocol;
pub mod node;
pub mod bio;

pub use protocol::HandoverData;
pub use node::P2PNode;
pub use bio::BioAntenna;
pub use bio_socket::start_bio_server;

pub use protocol::HandoverData;
pub use node::P2PNode;
use libp2p::{gossipsub, mdns, swarm::NetworkBehaviour};

#[derive(NetworkBehaviour)]
pub struct ArkheNetBehavior {
    pub gossipsub: gossipsub::Behaviour,
    pub mdns: mdns::tokio::Behaviour,
}
