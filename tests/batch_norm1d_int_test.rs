use mllayers::{Layer, BatchNorm1d};
use mlautograd::Tensor;

// @covers: BatchNorm1d::new
#[test]
fn test_batch_norm1d_struct_new_creates_with_two_params() {
    let bn = BatchNorm1d::new(4);
    assert_eq!(bn.parameters().len(), 2);
}

// @covers: BatchNorm1d::forward
#[test]
fn test_batch_norm1d_struct_forward_output_shape_matches_input() {
    let mut bn = BatchNorm1d::new(3);
    let input = Tensor::randn([4, 3]);
    let output = bn.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[4, 3]);
}

// @covers: BatchNorm1d::num_features
#[test]
fn test_batch_norm1d_struct_num_features_returns_configured_value() {
    let bn = BatchNorm1d::new(8);
    assert_eq!(bn.num_features(), 8);
}

// @covers: BatchNorm1d::is_training
#[test]
fn test_batch_norm1d_struct_is_training_true_by_default() {
    let bn = BatchNorm1d::new(4);
    assert!(bn.is_training());
}

// @covers: BatchNorm1d::eval
#[test]
fn test_batch_norm1d_struct_eval_sets_training_to_false() {
    let mut bn = BatchNorm1d::new(4);
    bn.eval();
    assert!(!bn.is_training());
}
