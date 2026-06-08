use crate::api::types::conv1d::Conv1d;
use crate::api::types::conv1d_builder::Conv1dBuilder;

impl Conv1dBuilder {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            dilation: 1,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    // @covers: new
    #[test]
    fn test_new_uses_default_stride_padding_dilation() {
        let layer = Conv1dBuilder::new(3, 8, 5).build();
        assert_eq!(layer.stride(), 1);
        assert_eq!(layer.padding(), 0);
        assert_eq!(layer.dilation(), 1);
    }

    // @covers: build
    #[test]
    fn test_build_applies_custom_config() {
        let layer = Conv1dBuilder::new(2, 4, 3).stride(2).padding(1).dilation(2).build();
        assert_eq!(layer.stride(), 2);
        assert_eq!(layer.padding(), 1);
        assert_eq!(layer.dilation(), 2);
    }
}
