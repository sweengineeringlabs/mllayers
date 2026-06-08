use mlautograd::MlError;

/// Error type for layer operations.
#[allow(dead_code)]
#[derive(Debug, thiserror::Error)]
pub enum LayerError {
    #[error("tensor operation failed: {0}")]
    TensorError(#[from] MlError),

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

/// Result alias using LayerError.
#[allow(dead_code)]
pub type LayerResult<T> = Result<T, LayerError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_error_invalid_config_formats_message() {
        let err = LayerError::InvalidConfig("rank must be > 0".to_string());
        assert!(err.to_string().contains("rank must be > 0"));
    }

    #[test]
    fn test_layer_error_shape_mismatch_formats_both_shapes() {
        let err = LayerError::ShapeMismatch {
            expected: vec![3, 4],
            actual: vec![2, 4],
        };
        let msg = err.to_string();
        assert!(msg.contains("3") || msg.contains("2"));
    }
}
