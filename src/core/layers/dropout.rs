use mlautograd::{BackwardOp, MlError, MlResult, Tensor, TapeEntry, tape};
use crate::api::traits::layer::Layer;
use crate::api::types::dropout::Dropout;
use rand::Rng;

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

    // @covers: new
    #[test]
    fn test_new_stores_probability_and_defaults_to_training() {
        let d = Dropout::new(0.3);
        assert!((d.p() - 0.3).abs() < f32::EPSILON);
        assert!(d.is_training());
    }

    // @covers: forward
    #[test]
    fn test_forward_eval_mode_passes_through_unchanged() {
        let mut d = Dropout::new(0.5);
        d.eval();
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).expect("input");
        let output = d.forward(&input).expect("forward");
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    // @covers: eval
    #[test]
    fn test_eval_disables_training_mode() {
        let mut d = Dropout::new(0.5);
        d.eval();
        assert!(!d.is_training());
    }
}
