use mlautograd::Tensor;

/// Batch Normalization for 1D inputs.
pub struct BatchNorm1d {
    pub(crate) gamma: Tensor,
    pub(crate) beta: Tensor,
    pub(crate) running_mean: Vec<f32>,
    pub(crate) running_var: Vec<f32>,
    pub(crate) num_features: usize,
    pub(crate) eps: f32,
    pub(crate) momentum: f32,
    pub(crate) training: bool,
}
