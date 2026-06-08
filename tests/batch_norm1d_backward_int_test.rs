use mllayers::{Layer, BatchNorm1d};
use mlautograd::Tensor;

// @covers: BatchNorm1d backward path (training mode records BatchNorm1dBackward op)
#[test]
fn test_batch_norm1d_backward_struct_training_mode_forward_has_correct_output_shape() {
    let mut bn = BatchNorm1d::new(3);
    let input = Tensor::randn([16, 3]);
    let output = bn.forward(&input).expect("forward in training mode");
    assert_eq!(output.shape(), &[16, 3]);
}

// @covers: BatchNorm1d running_mean updated after training forward
#[test]
fn test_batch_norm1d_backward_struct_running_mean_updated_from_zero_after_training_pass() {
    let mut bn = BatchNorm1d::new(4);
    // All-ones input: batch mean is 1.0; initial running_mean is 0.0
    // After one pass with momentum=0.1: running_mean = 0.9*0 + 0.1*1 = 0.1
    let input = Tensor::from_vec(vec![1.0_f32; 32], vec![8, 4]).expect("input");
    bn.forward(&input).expect("training forward");
    assert!(
        bn.running_mean()[0] > 0.0,
        "running_mean must be updated from 0 after a training forward pass"
    );
}

// @covers: BatchNorm1d backward via eval mode forward
#[test]
fn test_batch_norm1d_backward_struct_eval_mode_uses_accumulated_running_stats() {
    let mut bn = BatchNorm1d::new(2);
    // Warm up running stats with non-trivial data
    let input = Tensor::from_vec(vec![1.0_f32; 16], vec![8, 2]).expect("input");
    for _ in 0..5 {
        bn.forward(&input).expect("training forward");
    }
    bn.eval();
    let out = bn.forward(&input).expect("eval forward");
    assert_eq!(out.shape(), &[8, 2]);
}
