// arkhe-axos-instaweb/src/instaweb/routing.rs
use crate::instaweb::hyperbolic::{HyperbolicNode, HyperbolicRouter};

pub struct GeodesicRouter {
    pub router: HyperbolicRouter,
}

impl GeodesicRouter {
    pub fn new(local: HyperbolicNode) -> Self {
        Self {
            router: HyperbolicRouter::new(local),
        }
    }

    pub fn find_path(&self, destination: &HyperbolicNode) -> Vec<String> {
        let mut path = Vec::new();
        path.push(self.router.local_node.id.clone());

        if let Some(next) = self.router.next_hop(destination) {
            path.push(next.id.clone());
        }

        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instaweb::hyperbolic::HyperbolicNode;

    #[test]
    fn test_pathfinding() {
        let local = HyperbolicNode::new("origin", 0.0, 0.0, 0.0);
        let mut georouter = GeodesicRouter::new(local);

        let n1 = HyperbolicNode::new("relay1", 0.2, 0.0, 0.0);
        georouter.router.add_neighbor(n1);

        let dest = HyperbolicNode::new("target", 0.9, 0.0, 0.0);
        let path = georouter.find_path(&dest);

        assert_eq!(path.len(), 2);
        assert_eq!(path[0], "origin");
        assert_eq!(path[1], "relay1");
    }
}
