# Implementing microGPT in MoonBit

*A Character-Level Transformer with Manual Autograd*

## 1. Overview

This document describes how to implement a minimal GPT-style language
model ("microGPT") in MoonBit. The implementation includes:

-   Character-level tokenization
-   Transformer architecture (RMSNorm + Multi-Head Attention + MLP)
-   Manual reverse-mode automatic differentiation
-   Adam optimizer
-   Autoregressive sampling

The goal is educational clarity rather than performance.

------------------------------------------------------------------------

## 2. System Architecture

The system consists of five major components:

1.  Data preprocessing\
2.  Tokenization and vocabulary\
3.  Model definition\
4.  Training loop\
5.  Inference (sampling)

------------------------------------------------------------------------

## 3. Data Pipeline

### 3.1 Input Format

-   Input file: `input.txt`
-   Each line represents a training document
-   Empty lines are ignored

Example:

    emma
    olivia
    noah

### 3.2 Document Representation

After reading:

    docs : Array[String]

Each element is treated as a separate training example.

------------------------------------------------------------------------

## 4. Tokenization

### 4.1 Vocabulary Construction

Collect all unique characters from the dataset.

Add a special token:

    BOS (Beginning Of Sequence) = 0

Vocabulary size becomes:

    vocab_size = number_of_unique_characters + 1

### 4.2 Mappings

-   `stoi : Char -> Int`
-   `itos : Int -> Char`

### 4.3 Token Sequence Construction

Each document is converted to:

    [BOS, c1, c2, ..., cn, BOS]

------------------------------------------------------------------------

## 5. Model Architecture

### 5.1 Hyperparameters

Typical small configuration:

    n_embd     = 16
    n_head     = 4
    n_layer    = 1
    block_size = 8
    head_dim   = n_embd / n_head

### 5.2 Components

#### (1) Token Embedding

    wte : [vocab_size, n_embd]

#### (2) Positional Embedding

    wpe : [block_size, n_embd]

#### (3) Transformer Block (repeated n_layer times)

Each block contains:

-   RMSNorm
-   Multi-Head Self-Attention
-   Residual connection
-   RMSNorm
-   MLP (ReLU² activation)
-   Residual connection

#### (4) Language Modeling Head

    lm_head : [vocab_size, n_embd]

------------------------------------------------------------------------

## 6. Forward Pass

For each token position:

### 6.1 Embedding

    x = wte[token_id] + wpe[position_id]

### 6.2 RMSNorm

    scale = (mean(x²) + eps)^(-1/2)
    x = x * scale

### 6.3 Multi-Head Attention

For each head:

1.  Compute projections:

        q = Wq x
        k = Wk x
        v = Wv x

2.  Store k and v in KV cache

3.  Attention scores:

        score_t = (q · k_t) / sqrt(head_dim)

4.  Softmax over previous positions

5.  Weighted sum of values

Concatenate heads and project with:

    Wo

### 6.4 MLP

    x -> fc1 -> ReLU -> square -> fc2

### 6.5 Output

    logits = lm_head(x)

------------------------------------------------------------------------

## 7. Loss Function

For each position:

    probs = softmax(logits)
    loss_t = -log(probs[target_token])

Final loss:

    loss = mean(loss_t over positions)

------------------------------------------------------------------------

## 8. Manual Autograd

### 8.1 Value Structure

Each scalar value contains:

    data
    grad
    prev (dependencies)
    backward (function)

### 8.2 Graph Construction

Operations like:

    +, -, *, /, exp, log, pow, relu

create new Value nodes and record dependencies.

### 8.3 Backward Pass

1.  Topologically sort the graph\
2.  Set `loss.grad = 1`\
3.  Traverse in reverse order\
4.  Accumulate gradients

------------------------------------------------------------------------

## 9. Optimization (Adam)

For each parameter:

    m = β1 m + (1-β1) g
    v = β2 v + (1-β2) g²

Bias-corrected:

    m̂ = m / (1 - β1^t)
    v̂ = v / (1 - β2^t)

Update rule:

    p -= lr * m̂ / (sqrt(v̂) + eps)

------------------------------------------------------------------------

## 10. Training Loop

For each step:

1.  Select document\
2.  Construct token sequence\
3.  Forward pass\
4.  Compute mean loss\
5.  Backward pass\
6.  Adam update\
7.  Print loss

------------------------------------------------------------------------

## 11. Inference (Sampling)

### 11.1 Procedure

1.  Start with:

        token_id = BOS

2.  For each position:

    -   Forward pass\
    -   Apply temperature scaling\
    -   Compute softmax\
    -   Sample from distribution\
    -   Stop if BOS generated

### 11.2 Temperature

-   \< 1.0 → more deterministic\

-   1.0 → more random

------------------------------------------------------------------------

## 12. Limitations

Facts:

-   Scalar-based implementation\
-   No tensor acceleration\
-   High computational cost

Interpretation:

-   Intended for education\
-   Suitable for studying gradient flow and Transformer internals

------------------------------------------------------------------------

## 13. Conclusion

This document outlines a minimal GPT-style model implemented in MoonBit
using manual autograd and a Transformer architecture.

Pipeline summary:

Embedding → Attention → MLP → Projection → Loss → Backward → Adam →
Sampling
