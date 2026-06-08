use crate::api::types::batch_norm1d::BatchNorm1d;

/// Builder for BatchNorm1d layers.
pub struct BatchNorm1dBuilder {
    pub(crate) num_features: usize,
    pub(crate) eps: f32,
    pub(crate) momentum: f32,
}

impl BatchNorm1dBuilder {
    pub fn new(num_features: usize) -> Self {
        Self { num_features, eps: 1e-5, momentum: 0.1 }
    }

    pub fn eps(mut self, eps: f32) -> Self { self.eps = eps; self }
    pub fn momentum(mut self, momentum: f32) -> Self { self.momentum = momentum; self }

    pub fn build(self) -> BatchNorm1d {
        BatchNorm1d::with_config(self.num_features, self.eps, self.momentum)
    }
}
