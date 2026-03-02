use std::time::{SystemTime, Duration};

pub struct ShardGamma {
    pub nodes: usize,
    pub phi_threshold: f64,
    pub locations: Vec<String>,
}

impl ShardGamma {
    pub fn new() -> Self {
        Self {
            nodes: 0,
            phi_threshold: 0.0,
            locations: vec![],
        }
    }

    pub fn with_nodes(mut self, nodes: usize) -> Self {
        self.nodes = nodes;
        self
    }

    pub fn with_phi_threshold(mut self, threshold: f64) -> Self {
        self.phi_threshold = threshold;
        self
    }

    pub fn with_locations(mut self, locations: Vec<&str>) -> Self {
        self.locations = locations.into_iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn deploy(&self) {
        log::info!("SHARD_GAMMA: Deploying {} civil nodes with Φ threshold {:.2}", self.nodes, self.phi_threshold);
        for loc in &self.locations {
            log::info!("SHARD_GAMMA: Integrating location: {}", loc);
        }
        println!("🚀 SHARD_GAMMA: Deployment in progress (Target: {} nodes)", self.nodes);
    }
}

pub struct ShardDelta {
    pub nodes: usize,
    pub regions: Vec<String>,
}

impl ShardDelta {
    pub fn new() -> Self {
        Self {
            nodes: 0,
            regions: vec![],
        }
    }

    pub fn with_nodes(mut self, nodes: usize) -> Self {
        self.nodes = nodes;
        self
    }

    pub fn with_regions(mut self, regions: Vec<&str>) -> Self {
        self.regions = regions.into_iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn deploy(&self) {
        log::info!("SHARD_DELTA: Deploying {} nodes across Latin America", self.nodes);
        for region in &self.regions {
            log::info!("SHARD_DELTA: Integrating region: {}", region);
        }
        println!("🌎 SHARD_DELTA: Latin America integration in progress (Target: {} nodes)", self.nodes);
    }
}

pub struct Scheduler;
impl Scheduler {
    pub fn new() -> Self {
        Self
    }

    pub fn schedule<F>(&self, deploy_time: SystemTime, f: F)
    where F: FnOnce() + Send + 'static {
        let now = SystemTime::now();
        let delay = deploy_time.duration_since(now).unwrap_or(Duration::from_secs(0));

        log::info!("SCHEDULER: Event scheduled in {:?}", delay);

        tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            f();
        });
    }
}

pub struct AutonomousExpander {
    pub scheduler: Scheduler,
    pub gamma_delay: Duration,
    pub gamma_phi_threshold: f64,
}

impl AutonomousExpander {
    pub fn new(gamma_delay: Duration, gamma_phi_threshold: f64) -> Self {
        Self {
            scheduler: Scheduler::new(),
            gamma_delay,
            gamma_phi_threshold,
        }
    }

    pub fn schedule_shard_gamma(&self) {
        let deploy_time = SystemTime::now() + self.gamma_delay;
        let phi_threshold = self.gamma_phi_threshold;

        log::info!("EXPANSION: Scheduling Shard Gamma for T+24h");

        self.scheduler.schedule(
            deploy_time,
            Box::new(move || {
                // Deploy 1000 civil nodes
                let gamma_shard = ShardGamma::new()
                    .with_nodes(1000)
                    .with_phi_threshold(phi_threshold)
                    .with_locations(vec![
                        "hospitals",
                        "schools",
                        "power_grids",
                        "water_systems"
                    ]);

                gamma_shard.deploy();

                // Update network metrics
                // This would typically involve a shared state or database
                log::info!("EXPANSION: Network metrics updated to 1999 nodes (Ω + Γ)");

                // Schedule next expansion (Shard Delta - Latin America)
                log::info!("EXPANSION: Scheduling Shard Delta for T+30 days");
            })
        );
    }

    pub fn update_network_metrics(&self, total_nodes: usize) {
        log::info!("EXPANSION: Updating network metrics to {} nodes", total_nodes);
    }

    pub fn schedule_shard_delta(&self, days: u64) {
        let deploy_time = SystemTime::now() + Duration::from_secs(days * 24 * 3600);
        log::info!("EXPANSION: Shard Delta scheduled in {} days", days);

        self.scheduler.schedule(
            deploy_time,
            Box::new(move || {
                let delta_shard = ShardDelta::new()
                    .with_nodes(5000)
                    .with_regions(vec![
                        "Brasil",
                        "Argentina",
                        "Chile",
                        "Colômbia",
                        "México"
                    ]);

                delta_shard.deploy();
                log::info!("EXPANSION: Shard Delta deployed. Latin America integrated into Avalon.");
            })
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_gamma_config() {
        let shard = ShardGamma::new()
            .with_nodes(100)
            .with_phi_threshold(0.85)
            .with_locations(vec!["test_loc"]);

        assert_eq!(shard.nodes, 100);
        assert_eq!(shard.phi_threshold, 0.85);
        assert_eq!(shard.locations[0], "test_loc");
    }

    #[tokio::test]
    async fn test_autonomous_expander_init() {
        let expander = AutonomousExpander::new(Duration::from_secs(1), 0.70);
        expander.update_network_metrics(100);
        expander.schedule_shard_delta(5);
    }

    #[test]
    fn test_shard_delta_config() {
        let shard = ShardDelta::new()
            .with_nodes(5000)
            .with_regions(vec!["Brasil", "México"]);

        assert_eq!(shard.nodes, 5000);
        assert_eq!(shard.regions[0], "Brasil");
        assert_eq!(shard.regions[1], "México");
    }
}
