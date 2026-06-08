use mllayers::{Layer, Gelu};
use mlautograd::Tensor;

// @covers: Gelu::new
#[test]
fn test_gelu_struct_new_creates_parameterless_layer() {
    let gelu = Gelu::new();
    assert!(gelu.parameters().is_empty());
}

// @covers: Gelu::forward
#[test]
fn test_gelu_struct_forward_maps_zero_to_zero() {
    let mut gelu = Gelu::new();
    let input = Tensor::from_vec(vec![0.0], vec![1]).expect("input");
    let output = gelu.forward(&input).expect("forward");
    assert!(output.to_vec()[0].abs() < 1e-6);
}

// @covers: Gelu::forward
#[test]
fn test_gelu_struct_forward_output_shape_matches_input() {
    let mut gelu = Gelu::new();
    let input = Tensor::randn([2, 3]);
    let output = gelu.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[2, 3]);
}
