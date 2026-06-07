# mllayers Architecture

## Overview

`mllayers` is a single-crate neural network layer library for Rust. It defines the `Layer` trait and ships a standard set of differentiable layers — fully-connected, 1D convolution, normalization, activations, dropout, and a sequential container. All activation and normalization math is inlined (no external activation or normalization crate). Autograd support is provided by `mlautograd`: each layer's forward pass records ops on the thread-local gradient tape, and each layer registers the corresponding `BackwardOp` so that `tape::backward` can propagate gradients through the full model.

## Stakeholders & Concerns

| Stakeholder | Concerns |
|-------------|----------|
| Consumers (mltraining, mloptim) | Stable `Layer` trait contract (`parameters_mut` for weight updates, `forward` for inference and training), correct gradient propagation through composed layers |
| Maintainers | Adding new layer types without breaking the `Layer` trait, keeping activation math co-located with its backward op, clear builder pattern for layers with many config options |
| Library end users | Compose layers into models with `Sequential`, run `model.forward(&input)`, hand `model.parameters_mut()` to an optimizer |

## Component Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                           mllayers                               │
│                                                                  │
│  ┌──────────┐                                                    │
│  │  layer   │  Layer trait (forward / parameters / parameter_count) │
│  └────┬─────┘                                                    │
│       │ implemented by all layer structs                         │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                        layers/                          │     │
│  │                                                         │     │
│  │  ┌──────────┐  ┌─────────────────┐  ┌───────────────┐  │     │
│  │  │  linear  │  │     conv1d      │  │  batch_norm   │  │     │
│  │  │          │  │  + conv1d_      │  │  + batch_norm │  │     │
│  │  │  Linear  │  │    builder      │  │    _builder   │  │     │
│  │  └──────────┘  └─────────────────┘  └───────────────┘  │     │
│  │                                                         │     │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │     │
│  │  │layer_norm│  │ dropout  │  │    sequential      │   │     │
│  │  │          │  │          │  │                    │   │     │
│  │  │LayerNorm │  │ Dropout  │  │   Sequential       │   │     │
│  │  └──────────┘  └──────────┘  └────────────────────┘   │     │
│  │                                                         │     │
│  │  ┌──────────────────────────────────────────────────┐   │     │
│  │  │                  activations/                    │   │     │
│  │  │  GELU  SiLU  ReLU  Sigmoid  Tanh                │   │     │
│  │  │  + backward ops for each                        │   │     │
│  │  └──────────────────────────────────────────────────┘   │     │
│  └─────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
         │                         │
         ▼                         ▼
    mlautograd               llmtensor
  (Tensor, tape,           (CoreTensor —
   BackwardOp)              numeric ops)
```

## Layer Responsibilities

| Module | Responsibility | Key Types | Dependencies |
|--------|---------------|-----------|--------------|
| `layer` | Defines the `Layer` trait: `forward`, `parameters`, `parameters_mut`, `parameter_count`. All layer structs implement this. | `Layer` | `mlautograd::Tensor` |
| `layers::linear` | Fully-connected affine transform (y = xW^T + b). Xavier weight initialization. Registers `AddBackward` and `MatMulBackward` on the tape. | `Linear` | `mlautograd`, `llmtensor` |
| `layers::conv1d` | 1D convolution with configurable stride, padding, and dilation. `Conv1dBuilder` enforces valid config at construction time. | `Conv1d`, `Conv1dBuilder` | `mlautograd`, `llmtensor` |
| `layers::batch_norm` | Batch normalization with running mean/variance. `BatchNorm1dBuilder` selects train vs. eval mode. | `BatchNorm1d`, `BatchNorm1dBuilder` | `mlautograd`, `llmtensor` |
| `layers::layer_norm` | Normalize over the last tensor dimension; scale and shift via learned gamma/beta. Math inlined — no external normalization dep. Used in the standard pre-norm transformer pattern. | `LayerNorm` | `mlautograd`, `llmtensor` |
| `layers::dropout` | Inverted dropout: scale kept elements by 1/(1-p) during training; identity during eval. | `Dropout` | `mlautograd`, `rand` |
| `layers::sequential` | Chains a `Vec<Box<dyn Layer>>`. Delegates `forward` through each layer in order; collects parameters from all children. | `Sequential` | `layer` |
| `layers::activations` | Element-wise activation functions. Each activation struct also carries its backward op, registered via `tape::record_op` on the forward call. Math for GELU/SiLU is inlined to match GPT-2/LLaMA numerics exactly. | `GELU`, `SiLU`, `ReLU`, `Sigmoid`, `Tanh` | `mlautograd` |

## Data Flow

```
  Caller
    │
    │  model.forward(&input)
    │
    ▼
┌────────────────┐
│   Sequential   │  iterates layers in order
└───────┬────────┘
        │ layer[0].forward(&x) → y0
        │ layer[1].forward(&y0) → y1
        │  ...
        │ layer[n].forward(&y_{n-1}) → output
        ▼
┌────────────────────────────────────────┐
│  each Layer::forward                   │
│  ├── numeric op on Tensor (llmtensor)  │
│  └── tape::record_op(BackwardOp)       │──▶ GradientTape (mlautograd)
└────────────────────────────────────────┘
        │
        │  output tensor returned to caller
        ▼
  loss = loss_fn(output, target)
        │
        │  tape::backward(&loss)          (called by mltraining)
        ▼
┌────────────────────────────────────────┐
│  GradientTape replays entries reversed │
│  ├── activation backward ops           │
│  ├── linear/conv backward ops          │
│  └── norm backward ops                 │
└───────────────┬────────────────────────┘
                │  gradients accumulated per TensorId
                ▼
  mloptim::Optimizer::step(model.parameters_mut())
                │
                │  update weights in-place
                ▼
            next iteration
```

## Design Decisions

**`Layer` as an object-safe trait.**
`Sequential` stores `Vec<Box<dyn Layer>>`, so `Layer` must be object-safe. `parameters` and `parameters_mut` return `Vec<&Tensor>` and `Vec<&mut Tensor>` respectively, keeping the trait free of associated types or generics that would break object safety.

**Builders for multi-option layers.**
`Conv1dBuilder` and `BatchNorm1dBuilder` enforce valid configurations (e.g., dilation must be positive, kernel size must be odd for same-padding) at construction, moving errors from runtime to build time.

**Inlined activation and normalization math.**
GELU, SiLU, and LayerNorm math is written inline rather than delegating to external crates. This ensures the numerics match GPT-2/LLaMA exactly (critical for weight compatibility), keeps the dependency graph minimal, and makes the backward ops straightforward to co-locate with their forward counterparts.

**Backward ops co-located with the layer.**
Each layer module contains or imports its own backward op structs. This makes it easy to audit the full differentiable contract of a layer in one place and avoids a separate backward-only crate.

**No dependency on `mloptim` or `mltraining`.**
`mllayers` is intentionally import-free of training infrastructure. Inference-only consumers can import `mllayers` alone and call `tape::no_grad` from `mlautograd` to skip tape recording entirely.

## Integration Points

| System | Integration | Notes |
|--------|-------------|-------|
| `mlautograd` | `Tensor` for parameter storage; `tape::record_op` / `BackwardOp` / `TapeEntry` for gradient recording; `tape::no_grad` for inference mode | required dep |
| `llmtensor` | Underlying numeric ops on `CoreTensor` (via `mlautograd::Tensor` wrapper) | transitive via `mlautograd` |
| `mltraining` | `Trainer` calls `model.forward`, `tape::backward`, and `model.parameters_mut` to drive the training loop | consumer |
| `mloptim` | `Optimizer::step` iterates `model.parameters_mut()` to apply gradient updates | consumer |

## See Also

- [README](../../README.md)
