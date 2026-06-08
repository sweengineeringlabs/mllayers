use mlautograd::Tensor;

/// Linear layer: y = x @ W^T + b
pub struct Linear {
    pub(crate) weight: Tensor,
    pub(crate) bias: Tensor,
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
}
