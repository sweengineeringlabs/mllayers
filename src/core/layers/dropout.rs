use mlautograd::{BackwardOp, MlError, MlResult, Tensor, TapeEntry, tape};
use crate::api::layer::Layer;
use rand::Rng;

/// Dropout layer with inverted dropout scaling.
pub struct Dropout {
    p: f32,
    training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1), got {}",
            p
        );
        Self { p, training: true }
    }

    pub fn train(&mut self) { self.training = true; }
    pub fn eval(&mut self) { self.training = false; }
    pub fn is_training(&self) -> bool { self.training }
    pub fn p(&self) -> f32 { self.p }
}

impl Layer for Dropout {
    fn forward(&mut self, input: &Tensor) -> MlResult<Tensor> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let scale = 1.0 / (1.0 - self.p);
        let numel = input.numel();
        let mut rng = rand::thread_rng();
        let mask_data: Vec<f32> = (0..numel)
            .map(|_| {
                if rng.r#gen::<f32>() >= self.p { scale } else { 0.0 }
            })
            .collect();
        let mask = Tensor::from_vec(mask_data, input.shape().to_vec())
            .map_err(MlError::TensorError)?;

        let output = input.mul_raw(&mask)?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(DropoutBackward),
                output_id: output.id(),
                input_ids: vec![input.id()],
                saved_tensors: vec![mask],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> { vec![] }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> { vec![] }
}

struct DropoutBackward;

impl BackwardOp for DropoutBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let mask = &saved[0];
        let grad_input = grad_output.mul_raw(mask).expect("dropout backward mul");
        vec![grad_input]
    }

    fn name(&self) -> &str { "DropoutBackward" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_new_stores_probability() {
        let d = Dropout::new(0.3);
        assert!((d.p() - 0.3).abs() < f32::EPSILON);
        assert!(d.is_training());
    }

    #[test]
    fn test_dropout_eval_mode_passes_through_unchanged() {
        let mut d = Dropout::new(0.5);
        d.eval();
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let output = d.forward(&input).unwrap();
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0]);
    }
}
