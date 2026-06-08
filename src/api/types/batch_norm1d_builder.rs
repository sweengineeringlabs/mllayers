/// Builder for BatchNorm1d layers.
pub struct BatchNorm1dBuilder {
    pub(crate) num_features: usize,
    pub(crate) eps: f32,
    pub(crate) momentum: f32,
}
