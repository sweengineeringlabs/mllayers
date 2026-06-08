/// SiLU (Swish) activation function.
pub struct Silu;

impl Silu {
    pub fn new() -> Self { Self }
}

impl Default for Silu {
    fn default() -> Self { Self::new() }
}
