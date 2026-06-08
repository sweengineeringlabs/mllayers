use mlautograd::{MlResult, Tensor, TapeEntry, tape};
use mlautograd::gradient::tanh::TanhBackward;
use crate::api::traits::layer::Layer;
use crate::api::types::activations::Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Tanh {
    fn forward(&mut self, input: &Tensor) -> MlResult<Tensor> {
        let output = Tensor::new(input.inner().tanh(), false);

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(TanhBackward),
                output_id: output.id(),
                input_ids: vec![input.id()],
                saved_tensors: vec![output.clone()],
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
        let mut tanh_layer = Tanh::new();
        let input = Tensor::from_vec(vec![0.0], vec![1]).expect("input");
        let output = tanh_layer.forward(&input).expect("forward");
        assert!(output.to_vec()[0].abs() < 1e-6);
    }

    // @covers: parameters
    #[test]
    fn test_parameters_returns_empty_vec() {
        let t = Tanh::new();
        assert!(t.parameters().is_empty());
    }
}
