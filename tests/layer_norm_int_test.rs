use mllayers::{Layer, LayerNorm};
use mlautograd::Tensor;

// @covers: LayerNorm::new
#[test]
fn test_layer_norm_struct_new_creates_with_two_params() {
    let ln = LayerNorm::new(vec![4]);
    assert_eq!(ln.parameters().len(), 2);
}

// @covers: LayerNorm::forward
#[test]
fn test_layer_norm_struct_forward_output_shape_matches_input() {
    let mut ln = LayerNorm::new(vec![3]);
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
        .expect("input");
    let output = ln.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[2, 3]);
}

// @covers: LayerNorm::normalized_shape
#[test]
fn test_layer_norm_struct_normalized_shape_returns_configured_value() {
    let ln = LayerNorm::new(vec![4, 8]);
    assert_eq!(ln.normalized_shape(), &[4, 8]);
}

// @covers: LayerNorm::eps
#[test]
fn test_layer_norm_struct_eps_default_is_near_zero() {
    let ln = LayerNorm::new(vec![4]);
    assert!(ln.eps() < 1e-4);
}
