use mllayers::{Layer, Silu};
use mlautograd::Tensor;

// @covers: Silu::new
#[test]
fn test_silu_struct_new_creates_parameterless_layer() {
    let silu = Silu::new();
    assert!(silu.parameters().is_empty());
}

// @covers: Silu::forward
#[test]
fn test_silu_struct_forward_maps_zero_to_zero() {
    let mut silu = Silu::new();
    let input = Tensor::from_vec(vec![0.0], vec![1]).expect("input");
    let output = silu.forward(&input).expect("forward");
    assert!(output.to_vec()[0].abs() < 1e-6);
}

// @covers: Silu::forward
#[test]
fn test_silu_struct_forward_output_shape_matches_input() {
    let mut silu = Silu::new();
    let input = Tensor::randn([2, 4]);
    let output = silu.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[2, 4]);
}
