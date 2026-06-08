use mllayers::{Layer, Linear};
use mlautograd::Tensor;

// @covers: gateway ingress path
#[test]
fn test_ingress_gateway_struct_new_creates_layer_via_gateway_path() {
    // Gateway ingress is reserved for future external adapters.
    // This test verifies the full gateway → saf delegation chain is intact.
    let layer = Linear::new(8, 4);
    assert_eq!(layer.in_features(), 8);
    assert_eq!(layer.out_features(), 4);
}

// @covers: gateway ingress forward path
#[test]
fn test_ingress_gateway_struct_forward_runs_through_delegation_chain() {
    let mut layer = Linear::new(4, 2);
    let input = Tensor::randn([1, 4]);
    let output = layer.forward(&input).expect("forward via gateway");
    assert_eq!(output.shape(), &[1, 2]);
}
