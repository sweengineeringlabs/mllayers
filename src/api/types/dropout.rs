/// Dropout layer with inverted dropout scaling.
pub struct Dropout {
    pub(crate) p: f32,
    pub(crate) training: bool,
}
