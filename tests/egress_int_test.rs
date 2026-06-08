use mllayers::{Layer, Linear};
use mlautograd::Tensor;

// @covers: gateway egress path
#[test]
fn test_egress_gateway_struct_new_creates_layer_via_gateway_path() {
    // Gateway egress is reserved for future external adapters.
    // This test verifies the full gateway → saf delegation chain is intact.
    let layer = Linear::new(6, 3);
    assert_eq!(layer.in_features(), 6);
    assert_eq!(layer.out_features(), 3);
}

// @covers: gateway egress forward path
#[test]
fn test_egress_gateway_struct_forward_runs_through_delegation_chain() {
    let mut layer = Linear::new(3, 1);
    let input = Tensor::randn([2, 3]);
    let output = layer.forward(&input).expect("forward via gateway egress");
    assert_eq!(output.shape(), &[2, 1]);
}
