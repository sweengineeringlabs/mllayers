use mlautograd::{BackwardOp, MlResult, Tensor, TapeEntry, tape};
use mlautograd::gradient::add::unbroadcast;
use crate::api::traits::layer::Layer;
use crate::api::types::dropout::Dropout;
use crate::api::types::lora_linear::LoraLinear;

impl LoraLinear {
    /// Create a LoraLinear with freshly initialised base weights.
    pub fn new(
        in_features: usize,
        out_features: usize,
        r: usize,
        alpha: f32,
        dropout_p: Option<f32>,
    ) -> Self {
        assert!(r > 0, "LoRA rank r must be > 0, got {}", r);

        let xavier_scale = (6.0 / (in_features + out_features) as f32).sqrt();
        let mut base_weight = Tensor::randn([out_features, in_features]);
        base_weight = base_weight.mul_scalar_raw(xavier_scale);
        base_weight.set_requires_grad(false);

        let mut base_bias = Tensor::zeros([out_features]);
        base_bias.set_requires_grad(false);

        let a_scale = (2.0 / in_features as f32).sqrt();
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

    /// Create a LoraLinear from existing frozen base weights.
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

        let a_scale = (2.0 / in_features as f32).sqrt();
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
        if let Some(d) = &mut self.dropout {
            d.train();
        }
    }

    pub fn eval(&mut self) {
        self.training = false;
        if let Some(d) = &mut self.dropout {
            d.eval();
        }
    }

    pub fn r(&self) -> usize { self.r }
    pub fn scale(&self) -> f32 { self.scale }
    pub fn in_features(&self) -> usize { self.in_features }
    pub fn out_features(&self) -> usize { self.out_features }
}

impl Layer for LoraLinear {
    fn forward(&mut self, input: &Tensor) -> MlResult<Tensor> {
        let weight_t = self.base_weight.transpose_raw(-1, -2)?;
        let mut base_out = input.matmul_raw(&weight_t)?;
        if let Some(ref bias) = self.base_bias {
            base_out = base_out.add_raw(bias)?;
        }

        let lora_input = match &mut self.dropout {
            Some(d) if self.training => d.forward(input)?,
            _ => input.clone(),
        };

        let h = lora_input.matmul_raw(&self.lora_a)?;
        let lora_out = h.matmul_raw(&self.lora_b)?.mul_scalar_raw(self.scale);

        let output = base_out.add_raw(&lora_out)?;

        if tape::is_recording() {
            tape::record_op(TapeEntry {
                backward_op: Box::new(LoraLinearBackward {
                    input_shape: lora_input.shape().to_vec(),
                    lora_a_shape: self.lora_a.shape().to_vec(),
                    lora_b_shape: self.lora_b.shape().to_vec(),
                    scale: self.scale,
                }),
                output_id: output.id(),
                input_ids: vec![lora_input.id(), self.lora_a.id(), self.lora_b.id()],
                saved_tensors: vec![
                    lora_input,
                    self.base_weight.clone(),
                    self.lora_a.clone(),
                    self.lora_b.clone(),
                ],
            });
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.lora_a, &self.lora_b]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.lora_a, &mut self.lora_b]
    }
}

struct LoraLinearBackward {
    input_shape: Vec<usize>,
    lora_a_shape: Vec<usize>,
    lora_b_shape: Vec<usize>,
    scale: f32,
}

impl BackwardOp for LoraLinearBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let lora_input = &saved[0];
        let base_weight = &saved[1];
        let lora_a = &saved[2];
        let lora_b = &saved[3];

        let lora_b_t = lora_b.transpose_raw(-1, -2).expect("lora_b^T");
        let lora_a_t = lora_a.transpose_raw(-1, -2).expect("lora_a^T");

        let grad_base = grad_output
            .matmul_raw(base_weight)
            .expect("grad_input base path");
        let grad_through_b = grad_output
            .matmul_raw(&lora_b_t)
            .expect("grad @ lora_b^T");
        let grad_lora_path = grad_through_b
            .matmul_raw(&lora_a_t)
            .expect("grad_input lora path")
            .mul_scalar_raw(self.scale);
        let grad_input = unbroadcast(
            &grad_base.add_raw(&grad_lora_path).expect("grad_input add"),
            &self.input_shape,
        );

        let h = lora_input.matmul_raw(lora_a).expect("h = input @ lora_a");
        let grad_lora_b = unbroadcast(
            &h.transpose_raw(-1, -2)
                .expect("h^T")
                .matmul_raw(grad_output)
                .expect("grad_lora_b matmul")
                .mul_scalar_raw(self.scale),
            &self.lora_b_shape,
        );

        let grad_lora_a = unbroadcast(
            &lora_input
                .transpose_raw(-1, -2)
                .expect("lora_input^T")
                .matmul_raw(&grad_through_b)
                .expect("grad_lora_a matmul")
                .mul_scalar_raw(self.scale),
            &self.lora_a_shape,
        );

        vec![grad_input, grad_lora_a, grad_lora_b]
    }

    fn name(&self) -> &str {
        "LoraLinearBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // @covers: new
    #[test]
    fn test_new_creates_correct_lora_matrix_shapes() {
        let layer = LoraLinear::new(8, 4, 2, 2.0, None);
        let params = layer.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), &[8, 2], "lora_a shape");
        assert_eq!(params[1].shape(), &[2, 4], "lora_b shape");
    }

    // @covers: parameters_mut
    #[test]
    fn test_parameters_mut_returns_only_lora_matrices_not_base_weights() {
        let mut layer = LoraLinear::new(8, 4, 2, 2.0, None);
        assert_eq!(layer.parameters_mut().len(), 2);
        assert_eq!(layer.parameter_count(), 8 * 2 + 2 * 4);
    }

    // @covers: forward
    #[test]
    fn test_forward_output_shape_matches_base_linear() {
        let mut layer = LoraLinear::new(8, 4, 2, 2.0, None);
        let input = Tensor::randn([3, 8]);
        let output = layer.forward(&input).expect("forward");
        assert_eq!(output.shape(), &[3, 4]);
    }

    #[test]
    fn test_initial_lora_contribution_is_zero_because_b_is_zeros() {
        let in_f = 4;
        let out_f = 3;
        let mut layer = LoraLinear::new(in_f, out_f, 2, 1.0, None);

        let base_w = layer.base_weight.clone();
        let base_b = layer.base_bias.clone().expect("base_bias");
        let input = Tensor::randn([2, in_f]);

        let expected = input
            .matmul_raw(&base_w.transpose_raw(-1, -2).expect("transpose"))
            .expect("matmul")
            .add_raw(&base_b)
            .expect("add");

        let actual = layer.forward(&input).expect("forward");

        let exp_v = expected.to_vec();
        let act_v = actual.to_vec();
        for (e, a) in exp_v.iter().zip(act_v.iter()) {
            assert!(
                (e - a).abs() < 1e-5,
                "initial output should equal frozen base: expected {e}, got {a}"
            );
        }
    }

    // @covers: eval
    #[test]
    fn test_eval_mode_with_dropout_is_deterministic() {
        let mut layer = LoraLinear::new(8, 4, 2, 2.0, Some(0.5));
        layer.eval();
        let input = Tensor::randn([2, 8]);
        let out1 = layer.forward(&input).expect("out1").to_vec();
        let out2 = layer.forward(&input).expect("out2").to_vec();
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "eval mode output must be deterministic: {a} vs {b}"
            );
        }
    }

    // @covers: from_pretrained
    #[test]
    fn test_from_pretrained_uses_provided_base_weights() {
        let mut base_w = Tensor::zeros([4, 8]);
        base_w.set_requires_grad(false);
        let mut layer = LoraLinear::from_pretrained(base_w, None, 2, 2.0, None);

        let input = Tensor::randn([2, 8]);
        let output = layer.forward(&input).expect("forward");
        for v in output.to_vec() {
            assert!(v.abs() < 1e-6, "output with zero base and zero lora_b must be zero");
        }
    }
}
