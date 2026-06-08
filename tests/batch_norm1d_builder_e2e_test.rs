use mllayers::BatchNorm1dBuilder;

// @covers: BatchNorm1dBuilder::new
#[test]
fn test_batch_norm1d_builder_struct_new_sets_num_features() {
    let bn = BatchNorm1dBuilder::new(16).build();
    assert_eq!(bn.num_features(), 16);
}

// @covers: BatchNorm1dBuilder::eps
#[test]
fn test_batch_norm1d_builder_struct_eps_overrides_default() {
    let bn = BatchNorm1dBuilder::new(4).eps(1e-3).build();
    assert!((bn.eps() - 1e-3).abs() < 1e-6);
}

// @covers: BatchNorm1dBuilder::momentum
#[test]
fn test_batch_norm1d_builder_struct_momentum_overrides_default() {
    let bn = BatchNorm1dBuilder::new(4).momentum(0.05).build();
    assert!((bn.momentum() - 0.05).abs() < 1e-6);
}
