use mlautograd::Tensor;

/// 1-D convolutional layer.
pub struct Conv1d {
    pub(crate) weight: Tensor,
    pub(crate) bias: Tensor,
    pub(crate) in_channels: usize,
    pub(crate) out_channels: usize,
    pub(crate) kernel_size: usize,
    pub(crate) stride: usize,
    pub(crate) padding: usize,
    pub(crate) dilation: usize,
}
