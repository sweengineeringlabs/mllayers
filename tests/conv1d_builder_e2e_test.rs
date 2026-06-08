use mllayers::Conv1dBuilder;

// @covers: Conv1dBuilder::new
#[test]
fn test_conv1d_builder_struct_new_sets_correct_channels_and_kernel() {
    let layer = Conv1dBuilder::new(2, 8, 3).build();
    assert_eq!(layer.in_channels(), 2);
    assert_eq!(layer.out_channels(), 8);
    assert_eq!(layer.kernel_size(), 3);
}

// @covers: Conv1dBuilder::stride
#[test]
fn test_conv1d_builder_struct_stride_overrides_default() {
    let layer = Conv1dBuilder::new(2, 4, 3).stride(2).build();
    assert_eq!(layer.stride(), 2);
}

// @covers: Conv1dBuilder::padding
#[test]
fn test_conv1d_builder_struct_padding_overrides_default() {
    let layer = Conv1dBuilder::new(2, 4, 3).padding(1).build();
    assert_eq!(layer.padding(), 1);
}
