use mllayers::{Layer, Sigmoid};
use mlautograd::Tensor;

// @covers: Sigmoid::new
#[test]
fn test_sigmoid_struct_new_creates_parameterless_layer() {
    let sigmoid = Sigmoid::new();
    assert!(sigmoid.parameters().is_empty());
}

// @covers: Sigmoid::forward
#[test]
fn test_sigmoid_struct_forward_maps_zero_to_half() {
    let mut sigmoid = Sigmoid::new();
    let input = Tensor::from_vec(vec![0.0], vec![1]).expect("input");
    let output = sigmoid.forward(&input).expect("forward");
    assert!((output.to_vec()[0] - 0.5).abs() < 1e-6);
}

// @covers: Sigmoid::forward
#[test]
fn test_sigmoid_struct_forward_large_positive_input_approaches_one() {
    let mut sigmoid = Sigmoid::new();
    let input = Tensor::from_vec(vec![10.0], vec![1]).expect("input");
    let output = sigmoid.forward(&input).expect("forward");
    assert!(output.to_vec()[0] > 0.99);
}
