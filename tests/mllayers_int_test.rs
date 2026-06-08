//! Integration tests for the mllayers public API.
//!
//! These tests exercise the full stack via the saf/ public surface, verifying
//! that all public types work end-to-end with mlautograd tensors.
//!
//! @covers: Layer
//! @covers: Linear
//! @covers: Conv1d
//! @covers: Conv1dBuilder
//! @covers: BatchNorm1d
//! @covers: BatchNorm1dBuilder
//! @covers: LayerNorm
//! @covers: Dropout
//! @covers: Sequential
//! @covers: LoraLinear
//! @covers: Gelu
//! @covers: Relu
//! @covers: Silu
//! @covers: Sigmoid
//! @covers: Tanh

use mllayers::{
    BatchNorm1d, BatchNorm1dBuilder, Conv1d, Conv1dBuilder, Dropout, Gelu, Layer, LayerNorm,
    Linear, LoraLinear, Relu, Silu, Sequential, Sigmoid, Tanh,
};
use mlautograd::Tensor;

// ─── Linear ──────────────────────────────────────────────────────────────────

#[test]
fn test_linear_forward_produces_correct_output_shape() {
    let mut layer = Linear::new(4, 3);
    let input = Tensor::randn([2, 4]);
    let output = layer.forward(&input).expect("linear forward");
    assert_eq!(output.shape(), &[2, 3]);
}

#[test]
fn test_linear_parameter_count_is_weight_plus_bias() {
    let layer = Linear::new(4, 3);
    assert_eq!(layer.parameter_count(), 4 * 3 + 3);
}

// ─── Conv1d ──────────────────────────────────────────────────────────────────

#[test]
fn test_conv1d_forward_output_shape_no_padding() {
    let mut layer = Conv1d::new(2, 4, 3);
    let input = Tensor::randn([1, 2, 10]);
    let output = layer.forward(&input).expect("conv1d forward");
    assert_eq!(output.shape(), &[1, 4, 8]);
}

#[test]
fn test_conv1d_builder_produces_configured_layer() {
    let mut layer = Conv1dBuilder::new(2, 4, 3).stride(2).padding(1).build();
    let input = Tensor::randn([1, 2, 10]);
    let output = layer.forward(&input).expect("conv1d builder forward");
    assert_eq!(output.shape(), &[1, 4, 5]);
}

// ─── BatchNorm1d ─────────────────────────────────────────────────────────────

#[test]
fn test_batch_norm1d_forward_output_shape_2d() {
    let mut bn = BatchNorm1d::new(4);
    let input = Tensor::randn([8, 4]);
    let output = bn.forward(&input).expect("bn forward");
    assert_eq!(output.shape(), &[8, 4]);
}

#[test]
fn test_batch_norm1d_builder_applies_custom_eps() {
    let bn = BatchNorm1dBuilder::new(4).eps(1e-3).build();
    assert!((bn.eps() - 1e-3).abs() < 1e-10);
}

// ─── LayerNorm ───────────────────────────────────────────────────────────────

#[test]
fn test_layer_norm_forward_output_shape() {
    let mut ln = LayerNorm::new(vec![4]);
    let input = Tensor::randn([2, 4]);
    let output = ln.forward(&input).expect("layer_norm forward");
    assert_eq!(output.shape(), &[2, 4]);
}

// ─── Dropout ─────────────────────────────────────────────────────────────────

#[test]
fn test_dropout_eval_mode_passes_through() {
    let mut d = Dropout::new(0.5);
    d.eval();
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).expect("input");
    let output = d.forward(&input).expect("dropout forward");
    assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0]);
}

// ─── Sequential ──────────────────────────────────────────────────────────────

#[test]
fn test_sequential_chains_linear_and_relu() {
    let mut net = Sequential::new(vec![
        Box::new(Linear::new(4, 3)),
        Box::new(Relu::new()),
    ]);
    let input = Tensor::randn([2, 4]);
    let output = net.forward(&input).expect("sequential forward");
    assert_eq!(output.shape(), &[2, 3]);
}

#[test]
fn test_sequential_parameter_count_aggregates_sublayers() {
    let net = Sequential::new(vec![
        Box::new(Linear::new(4, 3)),
        Box::new(Linear::new(3, 2)),
    ]);
    assert_eq!(net.parameter_count(), (4 * 3 + 3) + (3 * 2 + 2));
}

// ─── LoraLinear ──────────────────────────────────────────────────────────────

#[test]
fn test_lora_linear_forward_output_shape() {
    let mut layer = LoraLinear::new(8, 4, 2, 2.0, None);
    let input = Tensor::randn([3, 8]);
    let output = layer.forward(&input).expect("lora forward");
    assert_eq!(output.shape(), &[3, 4]);
}

#[test]
fn test_lora_linear_only_exposes_lora_params() {
    let layer = LoraLinear::new(8, 4, 2, 2.0, None);
    assert_eq!(layer.parameter_count(), 8 * 2 + 2 * 4);
}

// ─── Activations ─────────────────────────────────────────────────────────────

#[test]
fn test_gelu_forward_zero_input_gives_zero_output() {
    let mut g = Gelu::new();
    let input = Tensor::from_vec(vec![0.0], vec![1]).expect("input");
    let output = g.forward(&input).expect("gelu forward");
    assert!(output.to_vec()[0].abs() < 1e-6);
}

#[test]
fn test_relu_forward_clamps_negatives_to_zero() {
    let mut r = Relu::new();
    let input = Tensor::from_vec(vec![-2.0, 0.0, 3.0], vec![3]).expect("input");
    let output = r.forward(&input).expect("relu forward");
    assert_eq!(output.to_vec(), vec![0.0, 0.0, 3.0]);
}

#[test]
fn test_silu_forward_zero_gives_zero() {
    let mut s = Silu::new();
    let input = Tensor::from_vec(vec![0.0], vec![1]).expect("input");
    let output = s.forward(&input).expect("silu forward");
    assert!(output.to_vec()[0].abs() < 1e-6);
}

#[test]
fn test_sigmoid_forward_zero_gives_half() {
    let mut s = Sigmoid::new();
    let input = Tensor::from_vec(vec![0.0], vec![1]).expect("input");
    let output = s.forward(&input).expect("sigmoid forward");
    assert!((output.to_vec()[0] - 0.5).abs() < 1e-6);
}

#[test]
fn test_tanh_forward_zero_gives_zero() {
    let mut t = Tanh::new();
    let input = Tensor::from_vec(vec![0.0], vec![1]).expect("input");
    let output = t.forward(&input).expect("tanh forward");
    assert!(output.to_vec()[0].abs() < 1e-6);
}

// ─── mlautograd dependency integration ───────────────────────────────────────

#[test]
fn test_mlautograd_tensor_integration_with_linear() {
    // Verifies that Tensor from the mlautograd dep works end-to-end with Linear.
    let mut layer = Linear::new(3, 2);
    let input = Tensor::from_vec(vec![1.0, 0.0, -1.0], vec![1, 3]).expect("input");
    let result = layer.forward(&input);
    assert!(result.is_ok(), "forward must succeed with valid input");
    assert_eq!(result.expect("output").shape(), &[1, 2]);
}
