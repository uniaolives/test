pub struct CoaxNetwork {
    pub hfc_nodes: Vec<HfcNode>,
    pub docsis_version: String,
}

pub struct HfcNode {
    pub id: String,
    pub frequency_range_mhz: f64,
}
