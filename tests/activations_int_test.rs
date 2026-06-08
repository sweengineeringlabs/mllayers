use mllayers::{Layer, Gelu, Relu, Silu, Sigmoid, Tanh};
use mlautograd::Tensor;

// @covers: activations module re-exports
#[test]
fn test_activations_gelu_struct_is_accessible_and_functional() {
    let mut g = Gelu::new();
    let out = g.forward(&Tensor::from_vec(vec![0.0], vec![1]).expect("t")).expect("fwd");
    assert!(out.to_vec()[0].abs() < 1e-6);
}

#[test]
fn test_activations_relu_struct_is_accessible_and_functional() {
    let mut r = Relu::new();
    let out = r.forward(&Tensor::from_vec(vec![-1.0], vec![1]).expect("t")).expect("fwd");
    assert_eq!(out.to_vec()[0], 0.0);
}

#[test]
fn test_activations_silu_struct_is_accessible_and_functional() {
    let mut s = Silu::new();
    let out = s.forward(&Tensor::from_vec(vec![0.0], vec![1]).expect("t")).expect("fwd");
    assert!(out.to_vec()[0].abs() < 1e-6);
}

#[test]
fn test_activations_sigmoid_struct_is_accessible_and_functional() {
    let mut sig = Sigmoid::new();
    let out = sig.forward(&Tensor::from_vec(vec![0.0], vec![1]).expect("t")).expect("fwd");
    assert!((out.to_vec()[0] - 0.5).abs() < 1e-6);
}

#[test]
fn test_activations_tanh_struct_is_accessible_and_functional() {
    let mut t = Tanh::new();
    let out = t.forward(&Tensor::from_vec(vec![0.0], vec![1]).expect("t")).expect("fwd");
    assert!(out.to_vec()[0].abs() < 1e-6);
}
