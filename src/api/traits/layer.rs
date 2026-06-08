use mlautograd::{MlResult, Tensor};

/// Core trait every neural-network layer must implement.
pub trait Layer {
    fn forward(&mut self, input: &Tensor) -> MlResult<Tensor>;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;

    fn parameter_count(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}
