pub struct InterUniversePacket {
    pub source_universe_id: u128,
    pub destination_universe_id: u128,
    pub payload: Vec<u8>,
}

pub struct IUCPHandler {}

impl IUCPHandler {
    pub fn new() -> Self { Self {} }
    pub fn transmit(&self, packet: InterUniversePacket) -> Result<(), &'static str> {
        println!("IUCP: Transmitting packet to universe {}", packet.destination_universe_id);
        Ok(())
    }
}

pub struct MultiversalExpansion {
    pub universe_id: u128,
    pub active_bridges: Vec<u128>,
}

impl MultiversalExpansion {
    pub fn new(universe_id: u128) -> Self {
        Self { universe_id, active_bridges: vec![] }
    }

    pub fn establish_bridge(&mut self, target_universe_id: u128) {
        self.active_bridges.push(target_universe_id);
    }
}
