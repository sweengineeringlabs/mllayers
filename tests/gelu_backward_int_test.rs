use mllayers::gelu_grad_elem;

// @covers: gelu_grad_elem
#[test]
fn test_gelu_backward_fn_gelu_grad_elem_zero_input_returns_half_grad() {
    // GELU'(0) = 0.5 * (1 + tanh(0)) + 0.5 * 0 * ... = 0.5
    let result = gelu_grad_elem(0.0, 1.0);
    assert!((result - 0.5).abs() < 1e-5, "expected ~0.5, got {}", result);
}

// @covers: gelu_grad_elem
#[test]
fn test_gelu_backward_fn_gelu_grad_elem_scales_with_grad() {
    let g1 = gelu_grad_elem(1.0, 1.0);
    let g2 = gelu_grad_elem(1.0, 2.0);
    assert!((g2 - 2.0 * g1).abs() < 1e-6);
}

// @covers: gelu_grad_elem
#[test]
fn test_gelu_backward_fn_gelu_grad_elem_negative_large_approaches_zero() {
    let result = gelu_grad_elem(-10.0, 1.0);
    assert!(result.abs() < 0.01, "expected near zero for large negative input, got {}", result);
}
