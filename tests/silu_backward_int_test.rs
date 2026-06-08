use mllayers::silu_grad_elem;

// @covers: silu_grad_elem
#[test]
fn test_silu_backward_fn_silu_grad_elem_zero_input_returns_half_grad() {
    // SiLU'(0) = 0.5 * (1 + 0 * ...) = 0.5
    let result = silu_grad_elem(0.0, 1.0);
    assert!((result - 0.5).abs() < 1e-5, "expected ~0.5, got {}", result);
}

// @covers: silu_grad_elem
#[test]
fn test_silu_backward_fn_silu_grad_elem_scales_with_grad() {
    let g1 = silu_grad_elem(1.0, 1.0);
    let g2 = silu_grad_elem(1.0, 3.0);
    assert!((g2 - 3.0 * g1).abs() < 1e-6);
}

// @covers: silu_grad_elem
#[test]
fn test_silu_backward_fn_silu_grad_elem_large_positive_approaches_one() {
    // SiLU'(x) → 1 as x → +∞
    let result = silu_grad_elem(20.0, 1.0);
    assert!(result > 0.99, "expected near 1 for large positive input, got {}", result);
}
