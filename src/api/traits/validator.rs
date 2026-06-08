use mlautograd::MlResult;

/// Validation contract for layer configurations.
pub trait Validator {
    fn validate(&self) -> MlResult<()>;
}
