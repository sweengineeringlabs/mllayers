use mllayers::{Layer, Dropout};
use mlautograd::Tensor;

// @covers: Dropout::new
#[test]
fn test_dropout_struct_new_creates_with_given_probability() {
    let d = Dropout::new(0.5);
    assert!((d.p() - 0.5).abs() < 1e-6);
}

// @covers: Dropout::forward
#[test]
fn test_dropout_struct_forward_eval_mode_passes_input_unchanged() {
    let mut d = Dropout::new(0.5);
    d.eval();
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).expect("input");
    let output = d.forward(&input).expect("forward");
    assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0]);
}

// @covers: Dropout::is_training
#[test]
fn test_dropout_struct_is_training_true_by_default() {
    let d = Dropout::new(0.3);
    assert!(d.is_training());
}

// @covers: Dropout::eval
#[test]
fn test_dropout_struct_eval_sets_training_to_false() {
    let mut d = Dropout::new(0.3);
    d.eval();
    assert!(!d.is_training());
}

// @covers: Dropout::parameters
#[test]
fn test_dropout_struct_parameters_returns_empty_vec() {
    let d = Dropout::new(0.5);
    assert!(d.parameters().is_empty());
}
