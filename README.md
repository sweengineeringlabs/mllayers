# mllayers

Neural network layers for Rust — Linear, Conv1d, LayerNorm, BatchNorm1d, GELU, SiLU, ReLU, Sigmoid, Tanh, Dropout, Sequential.

All GELU and SiLU math is inlined (no external activation crate dependency). All LayerNorm math is inlined (no external normalization crate dependency).

## Use cases

### ML: neural networks for vision, audio, tabular data
Compose `Linear`, `Conv1d`, `ReLU`, and `BatchNorm1d` layers into a `Sequential` model. Wire with `mloptim` and `mltraining` for a full training loop.

### LLM: transformer MLP blocks, embedding projections
Use `GELU` or `SiLU` activation (exact same math as GPT-2/LLaMA) inside feed-forward blocks built from `Linear` layers. `LayerNorm` implements the standard pre-norm pattern.

### Time-series: 1D convolutions for signal processing
`Conv1d` supports stride, padding, and dilation, making it suitable for dilated causal convolution stacks (WaveNet, TCN patterns).

### Inference-only: run layers without pulling in training infrastructure
Import `mllayers` alone — it does not depend on `mloptim` or `mltraining`. Set `tape::no_grad(|| ...)` from `mlautograd` to skip tape recording.

## Crate layout

| Module | Contents |
|---|---|
| `layer` | `Layer` trait |
| `layers::linear` | `Linear` (Xavier init, full backward) |
| `layers::conv1d` | `Conv1d` + `Conv1dBuilder` (stride/padding/dilation, full backward) |
| `layers::layer_norm` | `LayerNorm` (inlined math, full backward) |
| `layers::batch_norm` | `BatchNorm1d` + `BatchNorm1dBuilder` (train/eval modes, full backward) |
| `layers::activations` | `GELU`, `SiLU`, `ReLU`, `Sigmoid`, `Tanh` (all with backward ops) |
| `layers::dropout` | `Dropout` (inverted dropout, full backward) |
| `layers::sequential` | `Sequential` container |

## Quick start

```rust
use mllayers::{Linear, ReLU, Sequential, Layer};
use mlautograd::{Tensor, tape};

let mut model = Sequential::new(vec![
    Box::new(Linear::new(128, 64)),
    Box::new(ReLU::new()),
    Box::new(Linear::new(64, 10)),
]);

let input = Tensor::randn([32, 128]);
let output = model.forward(&input).unwrap();
assert_eq!(output.shape(), &[32, 10]);
```
