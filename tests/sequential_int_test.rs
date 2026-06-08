use mllayers::{Layer, Linear, Sequential, Relu};
use mlautograd::Tensor;

// @covers: Sequential::new
#[test]
fn test_sequential_struct_new_creates_with_correct_len() {
    let seq = Sequential::new(vec![
        Box::new(Linear::new(4, 3)),
        Box::new(Relu::new()),
    ]);
    assert_eq!(seq.len(), 2);
}

// @covers: Sequential::forward
#[test]
fn test_sequential_struct_forward_chains_layers_correctly() {
    let mut seq = Sequential::new(vec![
        Box::new(Linear::new(4, 3)),
        Box::new(Relu::new()),
    ]);
    let input = Tensor::randn([2, 4]);
    let output = seq.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[2, 3]);
}

// @covers: Sequential::is_empty
#[test]
fn test_sequential_struct_is_empty_returns_false_with_layers() {
    let seq = Sequential::new(vec![Box::new(Linear::new(4, 3))]);
    assert!(!seq.is_empty());
}

// @covers: Sequential::parameters
#[test]
fn test_sequential_struct_parameters_aggregates_sublayer_params() {
    let seq = Sequential::new(vec![
        Box::new(Linear::new(4, 3)),
        Box::new(Linear::new(3, 2)),
    ]);
    assert_eq!(seq.parameters().len(), 4);
}
