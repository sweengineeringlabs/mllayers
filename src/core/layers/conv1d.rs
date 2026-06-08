use mlautograd::{BackwardOp, MlError, MlResult, Tensor, TapeEntry, tape};
use crate::api::traits::layer::Layer;
use crate::api::types::conv1d::Conv1d;

impl Layer for Conv1d {
    fn forward(&mut self, input: &Tensor) -> MlResult<Tensor> {
        let input_shape = input.shape().to_vec();
        assert_eq!(input_shape.len(), 3, "Conv1d input must be 3-D");

        let batch = input_shape[0];
        let in_ch = input_shape[1];
        let length = input_shape[2];

        assert_eq!(in_ch, self.in_channels, "Conv1d: input channels mismatch");

        let out_length = (length + 2 * self.padding
            - self.dilation * (self.kernel_size - 1)
            - 1)
            / self.stride
            + 1;

        let input_data = input.to_vec();
        let weight_data = self.weight.to_vec();
        let bias_data = self.bias.to_vec();

        let mut output_data = vec![0.0f32; batch * self.out_channels * out_length];

        for b in 0..batch {
            for oc in 0..self.out_channels {
                for o in 0..out_length {
                    let mut sum = bias_data[oc];
                    for ic in 0..self.in_channels {
                        for k in 0..self.kernel_size {
                            let input_pos = (o * self.stride + k * self.dilation) as isize
                                - self.padding as isize;
                            if input_pos >= 0 && (input_pos as usize) < length {
                                let input_idx = b * in_ch * length + ic * length + input_pos as usize;
                                let weight_idx = oc * self.in_channels * self.kernel_size
                                    + ic * self.kernel_size + k;
                                sum += input_data[input_idx] * weight_data[weight_idx];
                            }
                        }
                    }
                    let output_idx = b * self.out_channels * out_length + oc * out_length + o;
                    output_data[output_idx] = sum;
                }
            }
        }

        let output = Tensor::from_vec(output_data, vec![batch, self.out_channels, out_length])
            .map_err(MlError::TensorError)?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(Conv1dBackward {
                    stride: self.stride,
                    padding: self.padding,
                    dilation: self.dilation,
                    in_channels: self.in_channels,
                    out_channels: self.out_channels,
                    kernel_size: self.kernel_size,
                    input_shape: input_shape.clone(),
                }),
                output_id: output.id(),
                input_ids: vec![input.id(), self.weight.id(), self.bias.id()],
                saved_tensors: vec![input.clone(), self.weight.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> { vec![&self.weight, &self.bias] }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> { vec![&mut self.weight, &mut self.bias] }
}

struct Conv1dBackward {
    stride: usize,
    padding: usize,
    dilation: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    input_shape: Vec<usize>,
}

impl BackwardOp for Conv1dBackward {
    fn name(&self) -> &str {
        std::any::type_name::<Self>().split("::").last().unwrap_or("Conv1dBackward")
    }

    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let input = &saved[0];
        let weight = &saved[1];

        let batch = self.input_shape[0];
        let in_ch = self.input_shape[1];
        let length = self.input_shape[2];

        let grad_shape = grad_output.shape().to_vec();
        let out_length = grad_shape[2];

        let grad_data = grad_output.to_vec();
        let input_data = input.to_vec();
        let weight_data = weight.to_vec();

        let mut grad_bias_data = vec![0.0f32; self.out_channels];
        for b in 0..batch {
            for oc in 0..self.out_channels {
                for o in 0..out_length {
                    let idx = b * self.out_channels * out_length + oc * out_length + o;
                    grad_bias_data[oc] += grad_data[idx];
                }
            }
        }

        let mut grad_weight_data =
            vec![0.0f32; self.out_channels * self.in_channels * self.kernel_size];
        for b in 0..batch {
            for oc in 0..self.out_channels {
                for o in 0..out_length {
                    let grad_idx = b * self.out_channels * out_length + oc * out_length + o;
                    let g = grad_data[grad_idx];
                    for ic in 0..self.in_channels {
                        for k in 0..self.kernel_size {
                            let input_pos = (o * self.stride + k * self.dilation) as isize
                                - self.padding as isize;
                            if input_pos >= 0 && (input_pos as usize) < length {
                                let input_idx = b * in_ch * length + ic * length + input_pos as usize;
                                let w_idx = oc * self.in_channels * self.kernel_size
                                    + ic * self.kernel_size + k;
                                grad_weight_data[w_idx] += g * input_data[input_idx];
                            }
                        }
                    }
                }
            }
        }

        let mut grad_input_data = vec![0.0f32; batch * in_ch * length];
        for b in 0..batch {
            for ic in 0..self.in_channels {
                for p in 0..length {
                    let mut sum = 0.0f32;
                    for oc in 0..self.out_channels {
                        for k in 0..self.kernel_size {
                            let numerator = p as isize + self.padding as isize
                                - (k * self.dilation) as isize;
                            if numerator >= 0 && numerator as usize % self.stride == 0 {
                                let o = numerator as usize / self.stride;
                                if o < out_length {
                                    let grad_idx = b * self.out_channels * out_length
                                        + oc * out_length + o;
                                    let w_idx = oc * self.in_channels * self.kernel_size
                                        + ic * self.kernel_size + k;
                                    sum += grad_data[grad_idx] * weight_data[w_idx];
                                }
                            }
                        }
                    }
                    let input_idx = b * in_ch * length + ic * length + p;
                    grad_input_data[input_idx] = sum;
                }
            }
        }

        let grad_input = Tensor::from_vec(grad_input_data, self.input_shape.clone())
            .expect("conv1d grad_input");
        let grad_weight = Tensor::from_vec(
            grad_weight_data,
            vec![self.out_channels, self.in_channels, self.kernel_size],
        ).expect("conv1d grad_weight");
        let grad_bias = Tensor::from_vec(grad_bias_data, vec![self.out_channels])
            .expect("conv1d grad_bias");

        vec![grad_input, grad_weight, grad_bias]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // @covers: forward
    #[test]
    fn test_forward_output_shape() {
        let mut layer = Conv1d::new(2, 4, 3);
        let input = Tensor::randn([1, 2, 10]);
        let output = layer.forward(&input).expect("forward");
        assert_eq!(output.shape(), &[1, 4, 8]);
    }

    // @covers: parameters
    #[test]
    fn test_parameters_returns_weight_and_bias() {
        let layer = Conv1d::new(3, 8, 5);
        assert_eq!(layer.parameters().len(), 2);
    }

    // @covers: parameters_mut
    #[test]
    fn test_parameters_mut_returns_weight_and_bias() {
        let mut layer = Conv1d::new(3, 8, 5);
        assert_eq!(layer.parameters_mut().len(), 2);
    }
}
