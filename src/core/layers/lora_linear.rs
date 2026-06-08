use mlautograd::{BackwardOp, MlResult, Tensor, TapeEntry, tape};
use mlautograd::gradient::add::unbroadcast;
use crate::api::traits::layer::Layer;
use crate::api::types::lora_linear::LoraLinear;

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

    fn parameters(&self) -> Vec<&Tensor> { vec![&self.lora_a, &self.lora_b] }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> { vec![&mut self.lora_a, &mut self.lora_b] }
}

struct LoraLinearBackward {
    input_shape: Vec<usize>,
    lora_a_shape: Vec<usize>,
    lora_b_shape: Vec<usize>,
    scale: f32,
}

impl BackwardOp for LoraLinearBackward {
    fn name(&self) -> &str {
        std::any::type_name::<Self>().split("::").last().unwrap_or("LoraLinearBackward")
    }

    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let lora_input = &saved[0];
        let base_weight = &saved[1];
        let lora_a = &saved[2];
        let lora_b = &saved[3];

        let lora_b_t = lora_b.transpose_raw(-1, -2).expect("lora_b^T");
        let lora_a_t = lora_a.transpose_raw(-1, -2).expect("lora_a^T");

        let grad_base = grad_output.matmul_raw(base_weight).expect("grad_input base path");
        let grad_through_b = grad_output.matmul_raw(&lora_b_t).expect("grad @ lora_b^T");
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
}

#[cfg(test)]
mod tests {
    use super::*;

    // @covers: forward
    #[test]
    fn test_forward_output_shape_matches_base_linear() {
        let mut layer = LoraLinear::new(8, 4, 2, 2.0, None);
        let input = Tensor::randn([3, 8]);
        let output = layer.forward(&input).expect("forward");
        assert_eq!(output.shape(), &[3, 4]);
    }

    // @covers: parameters
    #[test]
    fn test_parameters_returns_only_lora_matrices() {
        let layer = LoraLinear::new(8, 4, 2, 2.0, None);
        assert_eq!(layer.parameters().len(), 2);
    }

    // @covers: parameters_mut
    #[test]
    fn test_parameters_mut_returns_only_lora_matrices() {
        let mut layer = LoraLinear::new(8, 4, 2, 2.0, None);
        assert_eq!(layer.parameters_mut().len(), 2);
    }
}
