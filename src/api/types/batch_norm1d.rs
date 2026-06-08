use mlautograd::Tensor;

/// Batch Normalization for 1D inputs.
pub struct BatchNorm1d {
    pub(crate) gamma: Tensor,
    pub(crate) beta: Tensor,
    pub(crate) running_mean: Vec<f32>,
    pub(crate) running_var: Vec<f32>,
    pub(crate) num_features: usize,
    pub(crate) eps: f32,
    pub(crate) momentum: f32,
    pub(crate) training: bool,
}

impl BatchNorm1d {
    pub fn new(num_features: usize) -> Self {
        let mut gamma = Tensor::ones([num_features]);
        gamma.set_requires_grad(true);
        let mut beta = Tensor::zeros([num_features]);
        beta.set_requires_grad(true);
        Self {
            gamma,
            beta,
            running_mean: vec![0.0; num_features],
            running_var: vec![1.0; num_features],
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            training: true,
        }
    }

    pub fn with_config(num_features: usize, eps: f32, momentum: f32) -> Self {
        let mut bn = Self::new(num_features);
        bn.eps = eps;
        bn.momentum = momentum;
        bn
    }

    pub fn train(&mut self) { self.training = true; }
    pub fn eval(&mut self) { self.training = false; }
    pub fn is_training(&self) -> bool { self.training }
    pub fn eps(&self) -> f32 { self.eps }
    pub fn momentum(&self) -> f32 { self.momentum }
    pub fn num_features(&self) -> usize { self.num_features }
    pub fn running_mean(&self) -> &[f32] { &self.running_mean }
    pub fn running_var(&self) -> &[f32] { &self.running_var }
}
