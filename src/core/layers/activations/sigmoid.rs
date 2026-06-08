use mlautograd::{MlResult, Tensor, TapeEntry, tape};
use mlautograd::gradient::sigmoid::SigmoidBackward;
use crate::api::traits::layer::Layer;
use crate::api::types::activations::Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: &Tensor) -> MlResult<Tensor> {
        let output = Tensor::new(input.inner().sigmoid(), false);

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(SigmoidBackward),
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
    fn test_forward_maps_zero_to_half() {
        let mut sigmoid = Sigmoid::new();
        let input = Tensor::from_vec(vec![0.0], vec![1]).expect("input");
        let output = sigmoid.forward(&input).expect("forward");
        assert!((output.to_vec()[0] - 0.5).abs() < 1e-6);
    }

    // @covers: parameters
    #[test]
    fn test_parameters_returns_empty_vec() {
        let sig = Sigmoid::new();
        assert!(sig.parameters().is_empty());
    }
}
