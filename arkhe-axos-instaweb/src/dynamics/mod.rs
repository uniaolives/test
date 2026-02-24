#[derive(Default)]
pub struct State;

pub struct Geodesic {
    pub initial: State,
}

impl Geodesic {
    pub fn points(&self) -> impl Iterator<Item = crate::Point> {
        std::iter::once(crate::Point)
    }
}

pub struct SymplecticForm;
impl SymplecticForm {
    pub fn step(&self, _state: &State, _point: crate::Point) -> Result<State, crate::execution::Error> {
        Ok(State::default())
    }
}
