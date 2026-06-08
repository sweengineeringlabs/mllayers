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

impl Conv1d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let fan_in = in_channels * kernel_size;
        let fan_out = out_channels * kernel_size;
        let scale = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
        let mut weight = Tensor::randn([out_channels, in_channels, kernel_size]);
        weight = weight.mul_scalar_raw(scale);
        weight.set_requires_grad(true);
        let mut bias = Tensor::zeros([out_channels]);
        bias.set_requires_grad(true);
        Self { weight, bias, in_channels, out_channels, kernel_size, stride: 1, padding: 0, dilation: 1 }
    }

    pub fn with_stride(mut self, stride: usize) -> Self { self.stride = stride; self }
    pub fn with_padding(mut self, padding: usize) -> Self { self.padding = padding; self }
    pub fn with_dilation(mut self, dilation: usize) -> Self { self.dilation = dilation; self }

    pub fn in_channels(&self) -> usize { self.in_channels }
    pub fn out_channels(&self) -> usize { self.out_channels }
    pub fn kernel_size(&self) -> usize { self.kernel_size }
    pub fn stride(&self) -> usize { self.stride }
    pub fn padding(&self) -> usize { self.padding }
    pub fn dilation(&self) -> usize { self.dilation }
}
