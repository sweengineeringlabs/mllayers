use mllayers::{Layer, Linear};
use mlautograd::Tensor;

// @covers: Linear::new
#[test]
fn test_linear_struct_new_creates_weight_shape_out_by_in() {
    let layer = Linear::new(4, 3);
    assert_eq!(layer.parameters()[0].shape(), &[3, 4]);
}

// @covers: Linear::new
#[test]
fn test_linear_struct_new_creates_bias_shape_out() {
    let layer = Linear::new(4, 3);
    assert_eq!(layer.parameters()[1].shape(), &[3]);
}

// @covers: Linear::forward
#[test]
fn test_linear_struct_forward_output_shape_is_batch_by_out() {
    let mut layer = Linear::new(4, 3);
    let input = Tensor::randn([2, 4]);
    let output = layer.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[2, 3]);
}

// @covers: Linear::in_features
#[test]
fn test_linear_struct_in_features_returns_correct_value() {
    let layer = Linear::new(8, 4);
    assert_eq!(layer.in_features(), 8);
}

// @covers: Linear::out_features
#[test]
fn test_linear_struct_out_features_returns_correct_value() {
    let layer = Linear::new(8, 4);
    assert_eq!(layer.out_features(), 4);
}
