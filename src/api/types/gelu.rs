/// GELU activation function (approximate).
pub struct Gelu;

impl Gelu {
    pub fn new() -> Self { Self }
}

impl Default for Gelu {
    fn default() -> Self { Self::new() }
}
