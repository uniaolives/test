pub struct HyperbolicManifold;
impl HyperbolicManifold {
    pub fn geodesic(&self, initial: crate::dynamics::State, _target: crate::Point) -> crate::dynamics::Geodesic {
        crate::dynamics::Geodesic { initial }
    }
}
