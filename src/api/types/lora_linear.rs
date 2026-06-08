use mlautograd::Tensor;
use crate::api::types::dropout::Dropout;

/// Low-Rank Adaptation (LoRA) linear layer.
pub struct LoraLinear {
    pub(crate) base_weight: Tensor,
    pub(crate) base_bias: Option<Tensor>,
    pub(crate) lora_a: Tensor,
    pub(crate) lora_b: Tensor,
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
    pub(crate) r: usize,
    pub(crate) scale: f32,
    pub(crate) dropout: Option<Dropout>,
    pub(crate) training: bool,
}
