use mlautograd::{BackwardOp, Tensor};
use crate::api::layers::activations::silu_backward::silu_grad_elem;

pub(crate) struct SiluBackward;

impl BackwardOp for SiluBackward {
    fn name(&self) -> &str {
        std::any::type_name::<Self>().split("::").last().unwrap_or("SiluBackward")
    }

    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let input = &saved[0];
        let x_data = input.to_vec();
        let grad_data = grad_output.to_vec();

        let grad_input: Vec<f32> = x_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| silu_grad_elem(x, g))
            .collect();

        let result =
            Tensor::from_vec(grad_input, input.shape().to_vec()).expect("silu backward from_vec");
        vec![result]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // @covers: backward
    #[test]
    fn test_backward_output_shape_matches_input() {
        let op = SiluBackward;
        let input = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]).expect("input");
        let grad = Tensor::ones(vec![3]);
        let grads = op.backward(&grad, &[input]);
        assert_eq!(grads[0].shape(), &[3]);
    }
}
