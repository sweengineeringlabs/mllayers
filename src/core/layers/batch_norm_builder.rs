use super::batch_norm::BatchNorm1d;

pub struct BatchNorm1dBuilder {
    num_features: usize,
    eps: f32,
    momentum: f32,
}

impl BatchNorm1dBuilder {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
        }
    }

    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn build(self) -> BatchNorm1d {
        BatchNorm1d::with_config(self.num_features, self.eps, self.momentum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_new_uses_defaults() {
        let bn = BatchNorm1dBuilder::new(4).build();
        assert_eq!(bn.num_features(), 4);
        assert!((bn.eps() - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_builder_custom_eps_and_momentum() {
        let bn = BatchNorm1dBuilder::new(8).eps(1e-3).momentum(0.2).build();
        assert!((bn.eps() - 1e-3).abs() < 1e-10);
        assert!((bn.momentum() - 0.2).abs() < 1e-6);
    }
}
