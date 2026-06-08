use mlautograd::{MlResult, Tensor, TapeEntry, tape, ReLUBackward};
use crate::api::traits::layer::Layer;
use crate::api::types::activations::ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &Tensor) -> MlResult<Tensor> {
        let output = input.relu_raw();

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(ReLUBackward),
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
    fn test_forward_zeroes_negative_values() {
        let mut relu = ReLU::new();
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], vec![4]).expect("input");
        let output = relu.forward(&input).expect("forward");
        assert_eq!(output.to_vec(), vec![0.0, 0.0, 1.0, 2.0]);
    }

    // @covers: parameters
    #[test]
    fn test_parameters_returns_empty_vec() {
        let relu = ReLU::new();
        assert!(relu.parameters().is_empty());
    }
}
