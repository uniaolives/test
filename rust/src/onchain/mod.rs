#[derive(Debug, Clone, Copy)]
pub enum DeploymentTarget {
    Local,
    Mobile,
    SASC,
    EVM,
}
