use mlautograd::Tensor;
use crate::api::types::dropout::Dropout;

/// Low-Rank Adaptation (LoRA) linear layer.
pub struct LoraLinear {
    pub(crate) base_weight: Tensor,
    pub(crate) base_bias: Option<Tensor>,
    pub(crate) lora_a: Tensor,
    pub(crate) lora_b: Tensor,
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
    pub(crate) r: usize,
    pub(crate) scale: f32,
    pub(crate) dropout: Option<Dropout>,
    pub(crate) training: bool,
}

impl LoraLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        r: usize,
        alpha: f32,
        dropout_p: Option<f32>,
    ) -> Self {
        assert!(r > 0, "LoRA rank r must be > 0, got {}", r);
        let xavier_scale = (6.0_f32 / (in_features + out_features) as f32).sqrt();
        let mut base_weight = Tensor::randn([out_features, in_features]);
        base_weight = base_weight.mul_scalar_raw(xavier_scale);
        base_weight.set_requires_grad(false);
        let mut base_bias = Tensor::zeros([out_features]);
        base_bias.set_requires_grad(false);
        let a_scale = (2.0_f32 / in_features as f32).sqrt();
        let mut lora_a = Tensor::randn([in_features, r]);
        lora_a = lora_a.mul_scalar_raw(a_scale);
        lora_a.set_requires_grad(true);
        let mut lora_b = Tensor::zeros([r, out_features]);
        lora_b.set_requires_grad(true);
        Self {
            base_weight,
            base_bias: Some(base_bias),
            lora_a,
            lora_b,
            in_features,
            out_features,
            r,
            scale: alpha / r as f32,
            dropout: dropout_p.map(Dropout::new),
            training: true,
        }
    }

    pub fn from_pretrained(
        base_weight: Tensor,
        base_bias: Option<Tensor>,
        r: usize,
        alpha: f32,
        dropout_p: Option<f32>,
    ) -> Self {
        assert!(r > 0, "LoRA rank r must be > 0, got {}", r);
        let out_features = base_weight.shape()[0];
        let in_features = base_weight.shape()[1];
        let a_scale = (2.0_f32 / in_features as f32).sqrt();
        let mut lora_a = Tensor::randn([in_features, r]);
        lora_a = lora_a.mul_scalar_raw(a_scale);
        lora_a.set_requires_grad(true);
        let mut lora_b = Tensor::zeros([r, out_features]);
        lora_b.set_requires_grad(true);
        Self {
            base_weight,
            base_bias,
            lora_a,
            lora_b,
            in_features,
            out_features,
            r,
            scale: alpha / r as f32,
            dropout: dropout_p.map(Dropout::new),
            training: true,
        }
    }

    pub fn train(&mut self) {
        self.training = true;
        if let Some(d) = &mut self.dropout { d.train(); }
    }

    pub fn eval(&mut self) {
        self.training = false;
        if let Some(d) = &mut self.dropout { d.eval(); }
    }

    pub fn r(&self) -> usize { self.r }
    pub fn scale(&self) -> f32 { self.scale }
    pub fn in_features(&self) -> usize { self.in_features }
    pub fn out_features(&self) -> usize { self.out_features }
}
