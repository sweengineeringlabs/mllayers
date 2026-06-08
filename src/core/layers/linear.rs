use mlautograd::{BackwardOp, MlResult, Tensor, TapeEntry, tape};
use mlautograd::gradient::add::unbroadcast;
use crate::api::traits::layer::Layer;
use crate::api::types::linear::Linear;

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let scale = (6.0 / (in_features + out_features) as f32).sqrt();
        let mut weight = Tensor::randn([out_features, in_features]);
        weight = weight.mul_scalar_raw(scale);
        weight.set_requires_grad(true);

        let mut bias = Tensor::zeros([out_features]);
        bias.set_requires_grad(true);

        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Layer for Linear {
    fn forward(&mut self, input: &Tensor) -> MlResult<Tensor> {
        let weight_t = self.weight.transpose_raw(-1, -2)?;
        let matmul_result = input.matmul_raw(&weight_t)?;
        let output = matmul_result.add_raw(&self.bias)?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(LinearBackward {
                    input_shape: input.shape().to_vec(),
                    weight_shape: self.weight.shape().to_vec(),
                    bias_shape: self.bias.shape().to_vec(),
                }),
                output_id: output.id(),
                input_ids: vec![input.id(), self.weight.id(), self.bias.id()],
                saved_tensors: vec![input.clone(), self.weight.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }
}

struct LinearBackward {
    input_shape: Vec<usize>,
    weight_shape: Vec<usize>,
    bias_shape: Vec<usize>,
}

impl BackwardOp for LinearBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let input = &saved[0];
        let weight = &saved[1];

        let grad_input = grad_output.matmul_raw(weight).expect("linear grad_input");
        let grad_input = unbroadcast(&grad_input, &self.input_shape);

        let grad_output_t = grad_output.transpose_raw(-1, -2).expect("transpose grad");
        let grad_weight = grad_output_t.matmul_raw(input).expect("linear grad_weight");
        let grad_weight = unbroadcast(&grad_weight, &self.weight_shape);

        let grad_bias = unbroadcast(grad_output, &self.bias_shape);

        vec![grad_input, grad_weight, grad_bias]
    }

    fn name(&self) -> &str {
        "LinearBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // @covers: new
    #[test]
    fn test_new_creates_correct_parameter_shapes() {
        let layer = Linear::new(4, 3);
        assert_eq!(layer.in_features(), 4);
        assert_eq!(layer.out_features(), 3);
        let params = layer.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), &[3, 4]);
        assert_eq!(params[1].shape(), &[3]);
    }

    // @covers: forward
    #[test]
    fn test_forward_output_shape() {
        let mut layer = Linear::new(4, 3);
        let input = Tensor::randn([2, 4]);
        let output = layer.forward(&input).expect("forward");
        assert_eq!(output.shape(), &[2, 3]);
    }

    // @covers: parameters_mut
    #[test]
    fn test_parameters_mut_returns_weight_and_bias() {
        let mut layer = Linear::new(4, 3);
        assert_eq!(layer.parameters_mut().len(), 2);
    }
}
