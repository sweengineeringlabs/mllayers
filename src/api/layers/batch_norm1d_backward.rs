use mlautograd::Tensor;

/// Contract for the BatchNorm1d backward pass: computes per-channel gradients.
pub trait BatchNorm1dBackward {
    fn compute_channel_grad(
        &self,
        channel: usize,
        grad_output: &Tensor,
        x_hat: &Tensor,
        gamma: &Tensor,
    ) -> (f32, f32);
}
