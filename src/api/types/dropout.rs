/// Dropout layer with inverted dropout scaling.
pub struct Dropout {
    pub(crate) p: f32,
    pub(crate) training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!((0.0..1.0).contains(&p), "Dropout probability must be in [0, 1), got {}", p);
        Self { p, training: true }
    }

    pub fn train(&mut self) { self.training = true; }
    pub fn eval(&mut self) { self.training = false; }
    pub fn is_training(&self) -> bool { self.training }
    pub fn p(&self) -> f32 { self.p }
}
