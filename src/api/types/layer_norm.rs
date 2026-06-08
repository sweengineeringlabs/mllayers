use mlautograd::Tensor;

/// Layer Normalization.
pub struct LayerNorm {
    pub(crate) gamma: Tensor,
    pub(crate) beta: Tensor,
    pub(crate) normalized_shape: Vec<usize>,
    pub(crate) eps: f32,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        let total: usize = normalized_shape.iter().product();
        let mut gamma = Tensor::ones([total]);
        gamma.set_requires_grad(true);
        let mut beta = Tensor::zeros([total]);
        beta.set_requires_grad(true);
        Self { gamma, beta, normalized_shape, eps: 1e-5 }
    }

    pub fn with_eps(normalized_shape: Vec<usize>, eps: f32) -> Self {
        let mut ln = Self::new(normalized_shape);
        ln.eps = eps;
        ln
    }

    pub fn eps(&self) -> f32 { self.eps }
    pub fn normalized_shape(&self) -> &[usize] { &self.normalized_shape }
}
