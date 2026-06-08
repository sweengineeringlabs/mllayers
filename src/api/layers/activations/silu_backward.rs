/// Element-wise SiLU derivative: computes f'(x) * grad for the SiLU (Swish) activation.
pub fn silu_grad_elem(x: f32, grad: f32) -> f32 {
    let sig = 1.0 / (1.0 + (-x).exp());
    let d_silu = sig * (1.0 + x * (1.0 - sig));
    grad * d_silu
}
