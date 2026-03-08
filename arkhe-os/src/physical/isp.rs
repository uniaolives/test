pub struct InsidePlant {
    pub location: String, // "Jockey Club, Rio"
    pub servers: Vec<Server>,
    pub pqc_hsm: HardwareSecurityModule, // Para Dilithium3
}

impl InsidePlant {
    pub fn bootstrap(&self) {
        println!("🜏 Bootstrapping Inside Plant at {}...", self.location);
        for server in &self.servers {
            println!("  Starting server: {}", server.id);
        }
    }
}

pub struct Server {
    pub id: String,
    pub role: String,
}

pub struct HardwareSecurityModule {
    pub manufacturer: String,
    pub protocol: String,
}
