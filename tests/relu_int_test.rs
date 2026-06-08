use mllayers::{Layer, Relu};
use mlautograd::Tensor;

// @covers: Relu::new
#[test]
fn test_relu_struct_new_creates_parameterless_layer() {
    let relu = Relu::new();
    assert!(relu.parameters().is_empty());
}

// @covers: Relu::forward
#[test]
fn test_relu_struct_forward_zeroes_negative_values() {
    let mut relu = Relu::new();
    let input = Tensor::from_vec(vec![-1.0, 0.0, 2.0, -3.0], vec![4]).expect("input");
    let output = relu.forward(&input).expect("forward");
    assert_eq!(output.to_vec(), vec![0.0, 0.0, 2.0, 0.0]);
}

// @covers: Relu::forward
#[test]
fn test_relu_struct_forward_positive_values_pass_through_unchanged() {
    let mut relu = Relu::new();
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).expect("input");
    let output = relu.forward(&input).expect("forward");
    assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0]);
}
