# mllayers

> **TLDR:** Neural network layer library — Linear, Conv1d, normalization, activations, and more, with autograd tape support. See [Overview](docs/README.md) for details.

## Table of Contents
- [Quick Start](#quick-start)
- [API](#api)
- [Documentation](#documentation)

## Quick Start

```rust
use mllayers::{Layer, Linear, GELU, Sequential};
use mlautograd::{Tensor, tape};

let mut model = Sequential::new(vec![
    Box::new(Linear::new(128, 64)),
    Box::new(GELU::new()),
    Box::new(Linear::new(64, 1)),
]);

let input = Tensor::randn([32, 128]);
let output = model.forward(&input)?;
// output.shape() == [32, 1]
```

## API

| Type | Description |
|------|-------------|
| `Layer` | Implement to define a differentiable neural network layer (`forward`, `parameters`, `parameters_mut`, `parameter_count`) |
| `Linear` | Learnable affine transform: y = xW^T + b, with Xavier initialization |
| `Conv1d` / `Conv1dBuilder` | 1D convolution for sequence and signal processing; supports stride, padding, dilation |
| `BatchNorm1d` / `BatchNorm1dBuilder` | Batch normalization with separate train/eval modes |
| `LayerNorm` | Normalize over the last dimension; inlined math, no external dep — standard pre-norm for transformers |
| `Dropout` | Inverted dropout; no-op at eval time |
| `Sequential` | Chain layers into a model; delegates `forward` and `parameters_mut` across all children |
| `GELU` / `SiLU` | Modern activations used in LLMs and vision models (GPT-2 / LLaMA exact math) |
| `ReLU` / `Sigmoid` / `Tanh` | Classic activations, all with registered backward ops |

## Documentation

- [Architecture](docs/3-design/architecture.md) - System design

## Related FRs

None — foundational crate, no feature requests.
