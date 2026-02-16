# microGPT in MoonBit (Current Implementation)

This document describes the implementation in this repository as it exists now.

## Acknowledgment

Shout-out to the original `microgpt.py` by Andrej Karpathy, which this project
is based on:

https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

## Overview

The project implements a small character-level GPT in MoonBit with:

- manual reverse-mode autograd,
- a tiny Transformer-like forward pass,
- Adam optimization,
- autoregressive sampling.

The implementation is intentionally scalar and educational.

## File Map

- `value.mbt`: scalar autograd engine
- `tokenizer.mbt`: character vocabulary + BOS tokenization
- `model.mbt`: model weights, forward pass, loss, train, sample
- `microgpt.mbt`: convenience API (`run_microgpt`)
- `cmd/main/main.mbt`: runnable demo

## Tokenization

`Tokenizer` is built from all unique chars in the provided `docs : Array[String]`.

- Characters are sorted.
- `bos` is assigned to `chars.length()`.
- `vocab_size = chars.length() + 1`.

Document encoding:

```
[BOS, c1, c2, ..., cn, BOS]
```

## Autograd Engine

The autograd core is trait-oriented plus operator overloading.

### Traits and Operators

- `AutogradEngine` defines non-operator primitives:
  - `zero`, `scalar`, `pow`, `log`, `exp`, `relu`, `backward`
- `Value` implements operator traits:
  - `Add`, `Sub`, `Mul`, `Div`, `Neg`

This enables model expressions such as:

```
acc = acc + w * x
loss_t = -probs[target].log()
```

### Value Graph

Each `Value` stores:

- `data : Double`
- `grad : Double`
- dependency edges (`child`, `local_grad`)

Every operation creates a new `Value` node with local derivatives. `backward()`:

1. builds topological order by DFS,
2. sets output grad to `1.0`,
3. accumulates gradients in reverse topological order.

## Model Architecture

Defaults (`default_config()`):

- `n_embd = 16`
- `n_head = 4`
- `n_layer = 1`
- `block_size = 16`

Parameters include:

- token embeddings `wte`
- positional embeddings `wpe`
- per-layer attention matrices (`wq`, `wk`, `wv`, `wo`)
- per-layer MLP matrices (`fc1`, `fc2`)
- output projection `lm_head`

All parameters are `Value` scalars initialized from Gaussian noise.

## Forward Pass

Per position:

1. Token + position embedding
2. RMSNorm
3. For each layer:
   - RMSNorm
   - multi-head causal self-attention via KV caches
   - residual add
   - RMSNorm
   - MLP (`fc1 -> relu -> fc2`)
   - residual add
4. Output logits via `lm_head`

Softmax and negative log-likelihood are built from `Value` operations.

## Optimization

Training uses Adam with:

- `beta1 = 0.85`
- `beta2 = 0.99`
- `eps = 1e-8`
- linearly decayed LR over total training steps

After each update, parameter grads are reset to `0.0`.

## Sampling

Sampling is autoregressive and cached per layer.

1. Start at `token_id = BOS`
2. Recompute logits at each position
3. Apply temperature-scaled softmax on raw logits data
4. Sample next token from probability mass
5. Stop when BOS is sampled or `block_size` is reached

## Determinism

`Model::new` seeds RNG using a fixed byte pattern, so identical inputs and config yield deterministic train/sample behavior.

## Public API

- `build_tokenizer(docs)`
- `Model::new(docs, config)`
- `Model::train(num_steps)`
- `Model::sample(num_samples, temperature)`
- `run_microgpt(docs, num_steps, num_samples, temperature, config)`

## Limits

- Scalar graph only (no tensor kernels)
- Educational performance profile
- Small model intended for learning and experimentation
