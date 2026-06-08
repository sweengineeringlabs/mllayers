use mllayers::{Layer, Conv1d};
use mlautograd::Tensor;

// @covers: Conv1d::new
#[test]
fn test_conv1d_struct_new_creates_with_two_params() {
    let layer = Conv1d::new(2, 4, 3);
    assert_eq!(layer.parameters().len(), 2);
}

// @covers: Conv1d::forward
#[test]
fn test_conv1d_struct_forward_output_shape_is_batch_out_length() {
    let mut layer = Conv1d::new(2, 4, 3);
    let input = Tensor::randn([1, 2, 10]);
    let output = layer.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[1, 4, 8]);
}

// @covers: Conv1d::in_channels
#[test]
fn test_conv1d_struct_in_channels_returns_configured_value() {
    let layer = Conv1d::new(3, 8, 5);
    assert_eq!(layer.in_channels(), 3);
}

// @covers: Conv1d::stride
#[test]
fn test_conv1d_struct_stride_defaults_to_one() {
    let layer = Conv1d::new(2, 4, 3);
    assert_eq!(layer.stride(), 1);
}

// @covers: Conv1d::with_padding
#[test]
fn test_conv1d_struct_with_padding_output_length_increases_by_two_padding() {
    let mut layer = Conv1d::new(2, 4, 3).with_padding(1);
    let input = Tensor::randn([1, 2, 10]);
    let output = layer.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[1, 4, 10]);
}
