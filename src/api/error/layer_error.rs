use mlautograd::MlError;

/// Error type for layer operations.
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
