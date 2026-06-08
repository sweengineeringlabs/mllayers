//! Basic layers example — demonstrates creating and running forward passes
//! through Linear, Relu, and Sequential layers.

use mllayers::{Layer, Linear, Relu, Sequential};
use mlautograd::Tensor;

fn main() {
    // Build a simple MLP: Linear(4 -> 3) -> Relu -> Linear(3 -> 2)
    let net = Sequential::new(vec![
        Box::new(Linear::new(4, 3)),
        Box::new(Relu::new()),
        Box::new(Linear::new(3, 2)),
    ]);

    let param_count = net.parameter_count();
    println!("Network parameter count: {param_count}");
    assert_eq!(param_count, 4 * 3 + 3 + 3 * 2 + 2); // weight + bias for each linear

    // Run a forward pass.
    let mut net = net;
    let input = Tensor::randn([1, 4]);
    let output = net.forward(&input).expect("forward pass");
    println!("Output shape: {:?}", output.shape());
    assert_eq!(output.shape(), &[1, 2]);

    println!("basic_layers: OK");
}
