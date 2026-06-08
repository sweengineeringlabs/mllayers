use mllayers::{Layer, Tanh};
use mlautograd::Tensor;

// @covers: Tanh::new
#[test]
fn test_tanh_struct_new_creates_parameterless_layer() {
    let t = Tanh::new();
    assert!(t.parameters().is_empty());
}

// @covers: Tanh::forward
#[test]
fn test_tanh_struct_forward_maps_zero_to_zero() {
    let mut t = Tanh::new();
    let input = Tensor::from_vec(vec![0.0], vec![1]).expect("input");
    let output = t.forward(&input).expect("forward");
    assert!(output.to_vec()[0].abs() < 1e-6);
}

// @covers: Tanh::forward
#[test]
fn test_tanh_struct_forward_large_positive_approaches_one() {
    let mut t = Tanh::new();
    let input = Tensor::from_vec(vec![10.0], vec![1]).expect("input");
    let output = t.forward(&input).expect("forward");
    assert!(output.to_vec()[0] > 0.99);
}
