use mlautograd::{BackwardOp, MlError, MlResult, Tensor, TapeEntry, tape};
use crate::layer::Layer;

/// Layer Normalization.
///
/// Normalizes over the last dimension of the input, then applies an affine
/// transform: `output = gamma * normalized + beta`.
///
/// Reference: Ba, Kiros, Hinton - "Layer Normalization" (2016)
pub struct LayerNorm {
    gamma: Tensor,
    beta: Tensor,
    normalized_shape: Vec<usize>,
    eps: f32,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        let total: usize = normalized_shape.iter().product();

        let mut gamma = Tensor::ones([total]);
        gamma.set_requires_grad(true);

        let mut beta = Tensor::zeros([total]);
        beta.set_requires_grad(true);

        Self {
            gamma,
            beta,
            normalized_shape,
            eps: 1e-5,
        }
    }

    pub fn with_eps(normalized_shape: Vec<usize>, eps: f32) -> Self {
        let mut ln = Self::new(normalized_shape);
        ln.eps = eps;
        ln
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }

    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }

    /// Inline layer norm forward compute.
    /// Returns (output_data, normalized_data).
    fn compute(
        data: &[f32],
        gamma: &[f32],
        beta: &[f32],
        last_dim: usize,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = data.len() / last_dim;
        let d = last_dim as f32;
        let mut normalized_data = vec![0.0f32; data.len()];
        let mut output_data = vec![0.0f32; data.len()];

        for i in 0..n {
            let start = i * last_dim;
            let row = &data[start..start + last_dim];

            let mean: f32 = row.iter().sum::<f32>() / d;
            let var: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / d;
            let inv_std = 1.0 / (var + eps).sqrt();

            for j in 0..last_dim {
                let norm_val = (row[j] - mean) * inv_std;
                normalized_data[start + j] = norm_val;
                output_data[start + j] = gamma[j] * norm_val + beta[j];
            }
        }

        (output_data, normalized_data)
    }
}

impl Layer for LayerNorm {
    fn forward(&mut self, input: &Tensor) -> MlResult<Tensor> {
        let shape = input.shape().to_vec();
        let ndim = shape.len();
        assert!(ndim >= 1, "LayerNorm input must have at least 1 dimension");

        let last_dim = shape[ndim - 1];
        let norm_size: usize = self.normalized_shape.iter().product();
        assert_eq!(
            last_dim, norm_size,
            "LayerNorm: last dim {} != normalized_shape product {}",
            last_dim, norm_size
        );

        let input_data = input.to_vec();
        let gamma_data = self.gamma.to_vec();
        let beta_data = self.beta.to_vec();

        let (output_data, normalized_data) =
            Self::compute(&input_data, &gamma_data, &beta_data, last_dim, self.eps);

        let output = Tensor::from_vec(output_data, shape.clone())
            .map_err(MlError::TensorError)?;
        let normalized = Tensor::from_vec(normalized_data, shape)
            .map_err(MlError::TensorError)?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(LayerNormBackward {
                    eps: self.eps,
                    last_dim,
                }),
                output_id: output.id(),
                input_ids: vec![input.id(), self.gamma.id(), self.beta.id()],
                saved_tensors: vec![input.clone(), normalized, self.gamma.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.gamma, &self.beta]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.gamma, &mut self.beta]
    }
}

struct LayerNormBackward {
    eps: f32,
    last_dim: usize,
}

impl BackwardOp for LayerNormBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let input = &saved[0];
        let x_hat = &saved[1];
        let gamma = &saved[2];

        let grad_data = grad_output.to_vec();
        let input_data = input.to_vec();
        let x_hat_data = x_hat.to_vec();
        let gamma_data = gamma.to_vec();

        let last_dim = self.last_dim;
        let d = last_dim as f32;
        let n = input_data.len() / last_dim;

        let mut grad_input_data = vec![0.0f32; input_data.len()];
        let mut grad_gamma_data = vec![0.0f32; last_dim];
        let mut grad_beta_data = vec![0.0f32; last_dim];

        for i in 0..n {
            let start = i * last_dim;
            for j in 0..last_dim {
                let idx = start + j;
                grad_gamma_data[j] += grad_data[idx] * x_hat_data[idx];
                grad_beta_data[j] += grad_data[idx];
            }
        }

        for i in 0..n {
            let start = i * last_dim;
            let end = start + last_dim;
            let row = &input_data[start..end];

            let mean: f32 = row.iter().sum::<f32>() / d;
            let var: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / d;
            let inv_std = 1.0 / (var + self.eps).sqrt();

            let dx_hat: Vec<f32> = (0..last_dim)
                .map(|j| grad_data[start + j] * gamma_data[j])
                .collect();

            let sum_dx_hat: f32 = dx_hat.iter().sum();
            let sum_dx_hat_x_hat: f32 = (0..last_dim)
                .map(|j| dx_hat[j] * x_hat_data[start + j])
                .sum();

            for j in 0..last_dim {
                grad_input_data[start + j] = inv_std / d
                    * (d * dx_hat[j]
                        - sum_dx_hat
                        - x_hat_data[start + j] * sum_dx_hat_x_hat);
            }
        }

        let grad_input = Tensor::from_vec(grad_input_data, input.shape().to_vec())
            .expect("layer_norm grad_input");
        let grad_gamma =
            Tensor::from_vec(grad_gamma_data, vec![last_dim]).expect("layer_norm grad_gamma");
        let grad_beta =
            Tensor::from_vec(grad_beta_data, vec![last_dim]).expect("layer_norm grad_beta");

        vec![grad_input, grad_gamma, grad_beta]
    }

    fn name(&self) -> &str {
        "LayerNormBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_new_creates_correct_params() {
        let ln = LayerNorm::new(vec![4]);
        assert_eq!(ln.normalized_shape(), &[4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_with_eps() {
        let ln = LayerNorm::with_eps(vec![4], 1e-3);
        assert!((ln.eps() - 1e-3).abs() < 1e-10);
    }

    #[test]
    fn test_layer_norm_forward_output_shape() {
        let mut ln = LayerNorm::new(vec![3]);
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let output = ln.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3]);
    }
}
