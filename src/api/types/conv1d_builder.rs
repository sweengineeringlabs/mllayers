use crate::api::types::conv1d::Conv1d;

/// Builder for Conv1d layers.
pub struct Conv1dBuilder {
    pub(crate) in_channels: usize,
    pub(crate) out_channels: usize,
    pub(crate) kernel_size: usize,
    pub(crate) stride: usize,
    pub(crate) padding: usize,
    pub(crate) dilation: usize,
}

impl Conv1dBuilder {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self { in_channels, out_channels, kernel_size, stride: 1, padding: 0, dilation: 1 }
    }

    pub fn stride(mut self, stride: usize) -> Self { self.stride = stride; self }
    pub fn padding(mut self, padding: usize) -> Self { self.padding = padding; self }
    pub fn dilation(mut self, dilation: usize) -> Self { self.dilation = dilation; self }

    pub fn build(self) -> Conv1d {
        Conv1d::new(self.in_channels, self.out_channels, self.kernel_size)
            .with_stride(self.stride)
            .with_padding(self.padding)
            .with_dilation(self.dilation)
    }
}
