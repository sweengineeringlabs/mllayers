use mllayers::Validator;
use mlautograd::{MlError, MlResult};

struct AlwaysValid;
impl Validator for AlwaysValid {
    fn validate(&self) -> MlResult<()> { Ok(()) }
}

struct AlwaysInvalid;
impl Validator for AlwaysInvalid {
    fn validate(&self) -> MlResult<()> {
        Err(MlError::InvalidConfig("validation failed".to_string()))
    }
}

// @covers: Validator::validate
#[test]
fn test_validator_trait_validate_returns_ok_when_valid() {
    let v = AlwaysValid;
    assert!(v.validate().is_ok());
}

// @covers: Validator::validate
#[test]
fn test_validator_trait_validate_returns_err_when_invalid() {
    let v = AlwaysInvalid;
    assert!(v.validate().is_err());
}
