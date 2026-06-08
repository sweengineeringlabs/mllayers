use mlautograd::{BackwardOp, MlError, MlResult, Tensor, TapeEntry, tape};
use crate::api::traits::layer::Layer;
use crate::api::types::batch_norm1d::BatchNorm1d;

impl Layer for BatchNorm1d {
    fn forward(&mut self, input: &Tensor) -> MlResult<Tensor> {
        let shape = input.shape().to_vec();
        let ndim = shape.len();
        assert!(ndim == 2 || ndim == 3, "BatchNorm1d: expected 2D or 3D input, got {}D", ndim);

        let batch_size = shape[0];
        let channels = shape[1];
        assert_eq!(channels, self.num_features, "BatchNorm1d: expected {} features, got {}", self.num_features, channels);

        let spatial_size = if ndim == 3 { shape[2] } else { 1 };
        let count = batch_size * spatial_size;
        let count_f = count as f32;

        let data = input.to_vec();
        let gamma_data = self.gamma.to_vec();
        let beta_data = self.beta.to_vec();
        let total = data.len();

        let mut output_data = vec![0.0f32; total];
        let mut x_hat_data = vec![0.0f32; total];
        let mut batch_mean = vec![0.0f32; channels];
        let mut batch_var_eps = vec![0.0f32; channels];

        if self.training {
            for c in 0..channels {
                let mut sum = 0.0f32;
                for b in 0..batch_size {
                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + c * spatial_size + s;
                        sum += data[idx];
                    }
                }
                batch_mean[c] = sum / count_f;
            }

            let mut batch_var = vec![0.0f32; channels];
            for c in 0..channels {
                let mut sum_sq = 0.0f32;
                let mean_c = batch_mean[c];
                for b in 0..batch_size {
                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + c * spatial_size + s;
                        let diff = data[idx] - mean_c;
                        sum_sq += diff * diff;
                    }
                }
                batch_var[c] = sum_sq / count_f;
                batch_var_eps[c] = batch_var[c] + self.eps;
            }

            for c in 0..channels {
                let inv_std = 1.0 / batch_var_eps[c].sqrt();
                let mean_c = batch_mean[c];
                let g = gamma_data[c];
                let b_val = beta_data[c];
                for b in 0..batch_size {
                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + c * spatial_size + s;
                        let norm_val = (data[idx] - mean_c) * inv_std;
                        x_hat_data[idx] = norm_val;
                        output_data[idx] = g * norm_val + b_val;
                    }
                }
            }

            let m = self.momentum;
            for c in 0..channels {
                self.running_mean[c] = (1.0 - m) * self.running_mean[c] + m * batch_mean[c];
                self.running_var[c] = (1.0 - m) * self.running_var[c] + m * batch_var[c];
            }
        } else {
            for c in 0..channels {
                batch_mean[c] = self.running_mean[c];
                batch_var_eps[c] = self.running_var[c] + self.eps;
            }
            for c in 0..channels {
                let inv_std = 1.0 / batch_var_eps[c].sqrt();
                let mean_c = batch_mean[c];
                let g = gamma_data[c];
                let b_val = beta_data[c];
                for b in 0..batch_size {
                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + c * spatial_size + s;
                        let norm_val = (data[idx] - mean_c) * inv_std;
                        x_hat_data[idx] = norm_val;
                        output_data[idx] = g * norm_val + b_val;
                    }
                }
            }
        }

        let output = Tensor::from_vec(output_data, shape.clone()).map_err(MlError::TensorError)?;
        let x_hat = Tensor::from_vec(x_hat_data, shape).map_err(MlError::TensorError)?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(BatchNorm1dBackward {
                    num_features: self.num_features,
                    batch_var_eps,
                    batch_size,
                    spatial_size,
                }),
                output_id: output.id(),
                input_ids: vec![input.id(), self.gamma.id(), self.beta.id()],
                saved_tensors: vec![input.clone(), x_hat, self.gamma.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> { vec![&self.gamma, &self.beta] }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> { vec![&mut self.gamma, &mut self.beta] }
}

struct BatchNorm1dBackward {
    num_features: usize,
    batch_var_eps: Vec<f32>,
    batch_size: usize,
    spatial_size: usize,
}

impl BackwardOp for BatchNorm1dBackward {
    fn name(&self) -> &str {
        std::any::type_name::<Self>().split("::").last().unwrap_or("BatchNorm1dBackward")
    }

    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let input = &saved[0];
        let x_hat = &saved[1];
        let gamma = &saved[2];

        let grad_data = grad_output.to_vec();
        let x_hat_data = x_hat.to_vec();
        let gamma_data = gamma.to_vec();

        let channels = self.num_features;
        let batch_size = self.batch_size;
        let spatial_size = self.spatial_size;
        let count = batch_size * spatial_size;
        let count_f = count as f32;

        let total = grad_data.len();
        let mut grad_input_data = vec![0.0f32; total];
        let mut grad_gamma_data = vec![0.0f32; channels];
        let mut grad_beta_data = vec![0.0f32; channels];

        for c in 0..channels {
            for b in 0..batch_size {
                for s in 0..spatial_size {
                    let idx = b * channels * spatial_size + c * spatial_size + s;
                    grad_beta_data[c] += grad_data[idx];
                    grad_gamma_data[c] += grad_data[idx] * x_hat_data[idx];
                }
            }
        }

        for c in 0..channels {
            let inv_std = 1.0 / self.batch_var_eps[c].sqrt();
            let g = gamma_data[c];

            let mut sum_dx_hat = 0.0f32;
            let mut sum_dx_hat_x_hat = 0.0f32;

            for b in 0..batch_size {
                for s in 0..spatial_size {
                    let idx = b * channels * spatial_size + c * spatial_size + s;
                    let dx_hat = grad_data[idx] * g;
                    sum_dx_hat += dx_hat;
                    sum_dx_hat_x_hat += dx_hat * x_hat_data[idx];
                }
            }

            for b in 0..batch_size {
                for s in 0..spatial_size {
                    let idx = b * channels * spatial_size + c * spatial_size + s;
                    let dx_hat = grad_data[idx] * g;
                    grad_input_data[idx] = inv_std / count_f
                        * (count_f * dx_hat - sum_dx_hat - x_hat_data[idx] * sum_dx_hat_x_hat);
                }
            }
        }

        let grad_input = Tensor::from_vec(grad_input_data, input.shape().to_vec())
            .expect("batch_norm grad_input");
        let grad_gamma = Tensor::from_vec(grad_gamma_data, vec![channels])
            .expect("batch_norm grad_gamma");
        let grad_beta = Tensor::from_vec(grad_beta_data, vec![channels])
            .expect("batch_norm grad_beta");

        vec![grad_input, grad_gamma, grad_beta]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // @covers: forward
    #[test]
    fn test_forward_output_shape_2d() {
        let mut bn = BatchNorm1d::new(3);
        let input = Tensor::randn([4, 3]);
        let output = bn.forward(&input).expect("forward");
        assert_eq!(output.shape(), &[4, 3]);
    }

    // @covers: parameters
    #[test]
    fn test_parameters_returns_gamma_and_beta() {
        let bn = BatchNorm1d::new(4);
        assert_eq!(bn.parameters().len(), 2);
    }

    // @covers: parameters_mut
    #[test]
    fn test_parameters_mut_returns_gamma_and_beta() {
        let mut bn = BatchNorm1d::new(4);
        assert_eq!(bn.parameters_mut().len(), 2);
    }
}
