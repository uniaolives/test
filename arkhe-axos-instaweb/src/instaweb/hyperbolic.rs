// arkhe-axos-instaweb/src/instaweb/hyperbolic.rs
use nalgebra::Vector3;

#[derive(Debug, Clone)]
pub struct HyperbolicNode {
    pub id: String,
    pub coords: Vector3<f64>, // Poincare ball model (x, y, z) where ||coords|| < 1
}

impl HyperbolicNode {
    pub fn new(id: &str, x: f64, y: f64, z: f64) -> Self {
        let coords = Vector3::new(x, y, z);
        assert!(coords.norm() < 1.0, "Coordinates must be in Poincare ball (norm < 1)");
        Self {
            id: id.to_string(),
            coords,
        }
    }

    /// Calculates hyperbolic distance between two points in the Poincare ball.
    pub fn distance(&self, other: &Self) -> f64 {
        let u = self.coords;
        let v = other.coords;
        let uv_norm_sq = (u - v).norm_squared();
        let u_norm_sq = u.norm_squared();
        let v_norm_sq = v.norm_squared();

        let acosh_val = 1.0 + (2.0 * uv_norm_sq) / ((1.0 - u_norm_sq) * (1.0 - v_norm_sq));
        acosh_val.acosh()
    }
}

pub struct HyperbolicRouter {
    pub local_node: HyperbolicNode,
    pub neighbors: Vec<HyperbolicNode>,
}

impl HyperbolicRouter {
    pub fn new(local_node: HyperbolicNode) -> Self {
        Self {
            local_node,
            neighbors: Vec::new(),
        }
    }

    pub fn add_neighbor(&mut self, neighbor: HyperbolicNode) {
        self.neighbors.push(neighbor);
    }

    /// Greedy routing: find the neighbor closest to the destination in hyperbolic space.
    pub fn next_hop(&self, destination: &HyperbolicNode) -> Option<&HyperbolicNode> {
        self.neighbors
            .iter()
            .min_by(|a, b| {
                let da = a.distance(destination);
                let db = b.distance(destination);
                da.partial_cmp(&db).unwrap()
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperbolic_distance() {
        let n1 = HyperbolicNode::new("1", 0.0, 0.0, 0.0);
        let n2 = HyperbolicNode::new("2", 0.5, 0.0, 0.0);
        let dist = n1.distance(&n2);
        assert!(dist > 0.5); // Hyperbolic distance is always larger than Euclidean
    }

    #[test]
    fn test_greedy_routing() {
        let local = HyperbolicNode::new("local", 0.0, 0.0, 0.0);
        let mut router = HyperbolicRouter::new(local);

        let n1 = HyperbolicNode::new("n1", 0.1, 0.0, 0.0);
        let n2 = HyperbolicNode::new("n2", -0.1, 0.0, 0.0);
        router.add_neighbor(n1.clone());
        router.add_neighbor(n2.clone());

        let dest = HyperbolicNode::new("dest", 0.8, 0.0, 0.0);
        let next = router.next_hop(&dest).unwrap();
        assert_eq!(next.id, "n1");
    }
}
