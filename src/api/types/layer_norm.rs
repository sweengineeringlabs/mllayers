use mlautograd::Tensor;

/// Layer Normalization.
pub struct LayerNorm {
    pub(crate) gamma: Tensor,
    pub(crate) beta: Tensor,
    pub(crate) normalized_shape: Vec<usize>,
    pub(crate) eps: f32,
}
