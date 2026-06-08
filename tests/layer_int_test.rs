use mllayers::{Layer, Linear};
use mlautograd::Tensor;

// @covers: Layer::forward
#[test]
fn test_layer_trait_forward_returns_tensor_with_correct_shape() {
    let mut layer = Linear::new(4, 2);
    let input = Tensor::randn([3, 4]);
    let output = layer.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[3, 2]);
}

// @covers: Layer::parameters
#[test]
fn test_layer_trait_parameters_returns_learnable_tensors() {
    let layer = Linear::new(4, 2);
    assert_eq!(layer.parameters().len(), 2);
}

// @covers: Layer::parameters_mut
#[test]
fn test_layer_trait_parameters_mut_allows_in_place_update() {
    let mut layer = Linear::new(4, 2);
    let params = layer.parameters_mut();
    assert_eq!(params.len(), 2);
}
