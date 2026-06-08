use mlautograd::Tensor;

/// Linear layer: y = x @ W^T + b
pub struct Linear {
    pub(crate) weight: Tensor,
    pub(crate) bias: Tensor,
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let scale = (6.0_f32 / (in_features + out_features) as f32).sqrt();
        let mut weight = Tensor::randn([out_features, in_features]);
        weight = weight.mul_scalar_raw(scale);
        weight.set_requires_grad(true);
        let mut bias = Tensor::zeros([out_features]);
        bias.set_requires_grad(true);
        Self { weight, bias, in_features, out_features }
    }

    pub fn in_features(&self) -> usize { self.in_features }
    pub fn out_features(&self) -> usize { self.out_features }
}
