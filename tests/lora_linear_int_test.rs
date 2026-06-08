use mllayers::{Layer, LoraLinear};
use mlautograd::Tensor;

// @covers: LoraLinear::new
#[test]
fn test_lora_linear_struct_new_creates_with_two_lora_params() {
    let layer = LoraLinear::new(8, 4, 2, 2.0, None);
    assert_eq!(layer.parameters().len(), 2);
}

// @covers: LoraLinear::forward
#[test]
fn test_lora_linear_struct_forward_output_shape_matches_out_features() {
    let mut layer = LoraLinear::new(8, 4, 2, 2.0, None);
    let input = Tensor::randn([3, 8]);
    let output = layer.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[3, 4]);
}

// @covers: LoraLinear::r
#[test]
fn test_lora_linear_struct_r_returns_configured_rank() {
    let layer = LoraLinear::new(8, 4, 2, 2.0, None);
    assert_eq!(layer.r(), 2);
}

// @covers: LoraLinear::in_features
#[test]
fn test_lora_linear_struct_in_features_returns_configured_value() {
    let layer = LoraLinear::new(8, 4, 2, 2.0, None);
    assert_eq!(layer.in_features(), 8);
}

// @covers: LoraLinear::eval
#[test]
fn test_lora_linear_struct_eval_does_not_panic() {
    let mut layer = LoraLinear::new(8, 4, 2, 2.0, None);
    layer.eval();
    let input = Tensor::randn([1, 8]);
    layer.forward(&input).expect("forward in eval mode");
}
