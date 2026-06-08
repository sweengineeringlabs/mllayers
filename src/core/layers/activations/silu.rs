use mlautograd::{MlError, MlResult, Tensor, TapeEntry, tape};
use crate::api::traits::layer::Layer;
use crate::api::types::activations::SiLU;
use crate::core::layers::activations::silu_backward::SiLUBackward;

impl SiLU {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SiLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for SiLU {
    fn forward(&mut self, input: &Tensor) -> MlResult<Tensor> {
        let x_data = input.to_vec();

        let output_data: Vec<f32> = x_data
            .iter()
            .map(|&x| {
                let sig = 1.0 / (1.0 + (-x).exp());
                x * sig
            })
            .collect();

        let output = Tensor::from_vec(output_data, input.shape().to_vec())
            .map_err(MlError::TensorError)?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(SiLUBackward),
                output_id: output.id(),
                input_ids: vec![input.id()],
                saved_tensors: vec![input.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // @covers: forward
    #[test]
    fn test_forward_maps_zero_to_zero() {
        let mut silu = SiLU::new();
        let input = Tensor::from_vec(vec![0.0], vec![1]).expect("input");
        let output = silu.forward(&input).expect("forward");
        assert!(output.to_vec()[0].abs() < 1e-6);
    }

    // @covers: parameters
    #[test]
    fn test_parameters_returns_empty_vec() {
        let silu = SiLU::new();
        assert!(silu.parameters().is_empty());
    }
}
