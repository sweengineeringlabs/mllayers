/// Element-wise GELU derivative: computes f'(x) * grad for the approximate GELU.
pub fn gelu_grad_elem(x: f32, grad: f32) -> f32 {
    let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
    let x3 = x * x * x;
    let inner = sqrt_2_over_pi * (x + 0.044715 * x3);
    let tanh_inner = inner.tanh();
    let sech2 = 1.0 - tanh_inner * tanh_inner;
    let d_inner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x);
    let d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner;
    grad * d_gelu
}
