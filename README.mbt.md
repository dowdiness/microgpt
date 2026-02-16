# dowdiness/microgpt

Minimal character-level GPT in MoonBit with manual autograd.

Original inspiration: Andrej Karpathy's `microgpt.py` gist  
https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

## What this includes

- Character tokenizer with BOS token and encode/decode helpers
- Tiny GPT-style model (RMSNorm + causal self-attention + MLP)
- Scalar reverse-mode autograd engine
- Adam optimizer with linear LR decay
- Autoregressive sampling

## Current implementation notes

- Autograd is trait-based via `AutogradEngine` (`value.mbt`).
- `Value` also implements operator overloading (`Add`, `Sub`, `Mul`, `Div`, `Neg`) so model math uses `+ - * /`.
- Training and sampling are deterministic for the same inputs/config due fixed RNG seeding in `Model::new`.

## Run

```bash
moon run cmd/main
```

## Test

```bash
moon test
```

## Docs

- Implementation details: `docs/moonbit_microgpt.md`
