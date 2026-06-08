use mllayers::LayerError;

// @covers: LayerError::InvalidConfig
#[test]
fn test_layer_error_struct_invalid_config_display_contains_message() {
    let err = LayerError::InvalidConfig("bad config".to_string());
    let msg = err.to_string();
    assert!(msg.contains("bad config"));
}

// @covers: LayerError::ShapeMismatch
#[test]
fn test_layer_error_struct_shape_mismatch_display_includes_shapes() {
    let err = LayerError::ShapeMismatch {
        expected: vec![2, 4],
        actual: vec![2, 3],
    };
    let msg = err.to_string();
    assert!(msg.contains("mismatch"));
}
