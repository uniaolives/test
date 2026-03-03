use rust_decimal::Decimal;
use rust_decimal_macros::dec;

#[derive(Clone, Copy, Debug)]
pub struct HyperbolicCoord {
    pub r: Decimal,
    pub theta: Decimal,
    pub z: Decimal,
}

pub struct Node {
    pub id: u64,
    pub coord: HyperbolicCoord,
}

pub struct NodeCluster {
    pub nodes_vec: Vec<Node>,
}

impl NodeCluster {
    pub fn new(size: usize) -> Self {
        let mut nodes = Vec::new();
        for i in 0..size {
            nodes.push(Node {
                id: i as u64,
                coord: HyperbolicCoord {
                    r: dec!(0.5),
                    theta: dec!(0.0),
                    z: dec!(0.0),
                },
            });
        }
        Self { nodes_vec: nodes }
    }

    pub fn size(&self) -> usize {
        self.nodes_vec.len()
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes_vec
    }

    pub async fn aggregate_results<T>(&self, _results: Vec<T>) -> AggregatedResults {
        AggregatedResults
    }
}

pub struct AggregatedResults;

pub struct HSyncChannel;
