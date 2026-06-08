/// ReLU activation function.
pub struct Relu;

impl Relu {
    pub fn new() -> Self { Self }
}

impl Default for Relu {
    fn default() -> Self { Self::new() }
}
