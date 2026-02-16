///|
pub struct Config {
  n_embd : Int
  n_head : Int
  n_layer : Int
  block_size : Int
  init_std : Double
  learning_rate : Double
  beta1 : Double
  beta2 : Double
  eps_adam : Double
} derive(Show)

///|
pub fn default_config() -> Config {
  {
    n_embd: 16,
    n_head: 4,
    n_layer: 1,
    block_size: 16,
    init_std: 0.08,
    learning_rate: 0.01,
    beta1: 0.85,
    beta2: 0.99,
    eps_adam: 0.00000001,
  }
}

///|
pub struct TrainReport {
  losses : Array[Double]
}

///|
priv struct LayerWeights {
  attn_wq : Array[Array[Value]]
  attn_wk : Array[Array[Value]]
  attn_wv : Array[Array[Value]]
  attn_wo : Array[Array[Value]]
  mlp_fc1 : Array[Array[Value]]
  mlp_fc2 : Array[Array[Value]]
}

///|
priv struct Weights {
  wte : Array[Array[Value]]
  wpe : Array[Array[Value]]
  lm_head : Array[Array[Value]]
  layers : Array[LayerWeights]
}

///|
pub struct Model {
  config : Config
  tokenizer : Tokenizer
  priv docs : Array[String]
  priv weights : Weights
  priv params : Array[Value]
  priv m : Array[Double]
  priv v : Array[Double]
  priv rng : @random.Rand
}

///|
fn rand_normal(rand : @random.Rand) -> Double {
  let u1_raw = rand.double()
  let u1 = if u1_raw < 0.000000000001 { 0.000000000001 } else { u1_raw }
  let u2 = rand.double()
  let radius = (-2.0 * @math.ln(u1)).sqrt()
  let theta = 2.0 * @math.PI * u2
  radius * @math.cos(theta)
}

///|
fn make_matrix(
  rand : @random.Rand,
  nout : Int,
  nin : Int,
  std : Double,
) -> Array[Array[Value]] {
  Array::makei(nout, fn(_) {
    Array::makei(nin, fn(_) { value(rand_normal(rand) * std) })
  })
}

///|
fn append_matrix_params(
  params : Array[Value],
  matrix : Array[Array[Value]],
) -> Unit {
  for row in matrix {
    for p in row {
      params.push(p)
    }
  }
}

///|
fn vector_add(a : Array[Value], b : Array[Value]) -> Array[Value] {
  let out : Array[Value] = []
  for i = 0; i < a.length(); i = i + 1 {
    out.push(a[i] + b[i])
  }
  out
}

///|
fn linear(x : Array[Value], w : Array[Array[Value]]) -> Array[Value] {
  let out : Array[Value] = []
  for row in w {
    let mut acc = value(0.0)
    for i = 0; i < x.length(); i = i + 1 {
      acc = acc + row[i] * x[i]
    }
    out.push(acc)
  }
  out
}

///|
fn softmax(logits : Array[Value]) -> Array[Value] {
  if logits.length() == 0 {
    return []
  }

  let mut max_data = logits[0].data
  for i = 1; i < logits.length(); i = i + 1 {
    if logits[i].data > max_data {
      max_data = logits[i].data
    }
  }

  let max_value = value(max_data)
  let exps : Array[Value] = []
  for logit in logits {
    exps.push((logit - max_value).exp())
  }

  let total = sum_values(exps)
  let out : Array[Value] = []
  for e in exps {
    out.push(e / total)
  }
  out
}

///|
fn rmsnorm(x : Array[Value]) -> Array[Value] {
  let squares : Array[Value] = []
  for xi in x {
    squares.push(xi * xi)
  }
  let ms = sum_values(squares) / value(x.length().to_double())
  let scale = (ms + value(0.00001)).pow(-0.5)
  let out : Array[Value] = []
  for xi in x {
    out.push(xi * scale)
  }
  out
}

///|
fn Model::gpt_step(
  self : Model,
  token_id : Int,
  pos_id : Int,
  keys_cache : Array[Array[Array[Value]]],
  values_cache : Array[Array[Array[Value]]],
) -> Array[Value] {
  let tok_emb = self.weights.wte[token_id]
  let pos_emb = self.weights.wpe[pos_id]
  let mut x = vector_add(tok_emb, pos_emb)
  x = rmsnorm(x)

  let head_dim = self.config.n_embd / self.config.n_head
  let attn_scale = value(head_dim.to_double().sqrt())

  for li = 0; li < self.config.n_layer; li = li + 1 {
    let layer = self.weights.layers[li]

    // 1) Multi-head self-attention block.
    let x_residual = x
    x = rmsnorm(x)

    let q = linear(x, layer.attn_wq)
    let k = linear(x, layer.attn_wk)
    let v = linear(x, layer.attn_wv)

    keys_cache[li].push(k)
    values_cache[li].push(v)

    let x_attn : Array[Value] = []
    for h = 0; h < self.config.n_head; h = h + 1 {
      let hs = h * head_dim

      let attn_logits : Array[Value] = []
      for t = 0; t < keys_cache[li].length(); t = t + 1 {
        let mut score = value(0.0)
        for j = 0; j < head_dim; j = j + 1 {
          score = score + q[hs + j] * keys_cache[li][t][hs + j]
        }
        attn_logits.push(score / attn_scale)
      }

      let attn_weights = softmax(attn_logits)

      for j = 0; j < head_dim; j = j + 1 {
        let mut out_j = value(0.0)
        for t = 0; t < values_cache[li].length(); t = t + 1 {
          out_j = out_j + attn_weights[t] * values_cache[li][t][hs + j]
        }
        x_attn.push(out_j)
      }
    }

    x = linear(x_attn, layer.attn_wo)
    x = vector_add(x, x_residual)

    // 2) MLP block.
    let x_mlp_residual = x
    x = rmsnorm(x)
    x = linear(x, layer.mlp_fc1)
    let mlp_activated : Array[Value] = []
    for xi in x {
      mlp_activated.push(xi.relu())
    }
    x = linear(mlp_activated, layer.mlp_fc2)
    x = vector_add(x, x_mlp_residual)
  }

  linear(x, self.weights.lm_head)
}

///|
fn Model::loss_for_doc(self : Model, doc : String) -> Value {
  let tokens = self.tokenizer.encode_doc(doc)
  let n_raw = tokens.length() - 1
  let n = if n_raw < self.config.block_size {
    n_raw
  } else {
    self.config.block_size
  }

  let keys_cache : Array[Array[Array[Value]]] = Array::makei(
    self.config.n_layer,
    fn(_) { [] },
  )
  let values_cache : Array[Array[Array[Value]]] = Array::makei(
    self.config.n_layer,
    fn(_) { [] },
  )

  let losses : Array[Value] = []
  for pos_id = 0; pos_id < n; pos_id = pos_id + 1 {
    let token_id = tokens[pos_id]
    let target_id = tokens[pos_id + 1]
    let logits = self.gpt_step(token_id, pos_id, keys_cache, values_cache)
    let probs = softmax(logits)
    losses.push(-probs[target_id].log())
  }

  sum_values(losses) * value(1.0 / n.to_double())
}

///|
fn Model::adam_update(self : Model, step : Int, num_steps : Int) -> Unit {
  let lr_t = self.config.learning_rate *
    (1.0 - step.to_double() / num_steps.to_double())
  let t = (step + 1).to_double()

  for i = 0; i < self.params.length(); i = i + 1 {
    let p = self.params[i]
    let g = p.grad

    self.m[i] = self.config.beta1 * self.m[i] + (1.0 - self.config.beta1) * g
    self.v[i] = self.config.beta2 * self.v[i] +
      (1.0 - self.config.beta2) * g * g

    let m_hat = self.m[i] / (1.0 - @math.pow(self.config.beta1, t))
    let v_hat = self.v[i] / (1.0 - @math.pow(self.config.beta2, t))

    p.data = p.data - lr_t * m_hat / (v_hat.sqrt() + self.config.eps_adam)
    p.grad = 0.0
  }
}

///|
pub fn Model::new(docs : Array[String], config : Config) -> Model {
  let clean_docs = normalize_docs(docs)
  if clean_docs.is_empty() {
    abort("Model::new: docs must be non-empty")
  }
  if config.n_embd <= 0 ||
    config.n_head <= 0 ||
    config.n_layer <= 0 ||
    config.block_size <= 0 {
    abort("Model::new: all integer hyperparameters must be positive")
  }
  if config.n_embd % config.n_head != 0 {
    abort("Model::new: n_embd must be divisible by n_head")
  }

  let seed = Bytes::make(32, Int::to_byte(7))
  let rng = @random.Rand::chacha8(seed~)

  let shuffled_docs = clean_docs.copy()
  rng.shuffle(shuffled_docs.length(), fn(i, j) { shuffled_docs.swap(i, j) })

  let tokenizer = build_tokenizer(shuffled_docs)
  let vocab_size = tokenizer.vocab_size()

  let wte = make_matrix(rng, vocab_size, config.n_embd, config.init_std)
  let wpe = make_matrix(rng, config.block_size, config.n_embd, config.init_std)
  let lm_head = make_matrix(rng, vocab_size, config.n_embd, config.init_std)

  let layers = Array::makei(config.n_layer, fn(_) {
    {
      attn_wq: make_matrix(rng, config.n_embd, config.n_embd, config.init_std),
      attn_wk: make_matrix(rng, config.n_embd, config.n_embd, config.init_std),
      attn_wv: make_matrix(rng, config.n_embd, config.n_embd, config.init_std),
      attn_wo: make_matrix(rng, config.n_embd, config.n_embd, config.init_std),
      mlp_fc1: make_matrix(
        rng,
        4 * config.n_embd,
        config.n_embd,
        config.init_std,
      ),
      mlp_fc2: make_matrix(
        rng,
        config.n_embd,
        4 * config.n_embd,
        config.init_std,
      ),
    }
  })

  let weights : Weights = { wte, wpe, lm_head, layers }

  let params : Array[Value] = []
  append_matrix_params(params, wte)
  append_matrix_params(params, wpe)
  append_matrix_params(params, lm_head)
  for layer in layers {
    append_matrix_params(params, layer.attn_wq)
    append_matrix_params(params, layer.attn_wk)
    append_matrix_params(params, layer.attn_wv)
    append_matrix_params(params, layer.attn_wo)
    append_matrix_params(params, layer.mlp_fc1)
    append_matrix_params(params, layer.mlp_fc2)
  }

  {
    config,
    tokenizer,
    docs: shuffled_docs,
    weights,
    params,
    m: Array::make(params.length(), 0.0),
    v: Array::make(params.length(), 0.0),
    rng,
  }
}

///|
pub fn Model::num_params(self : Model) -> Int {
  self.params.length()
}

///|
pub fn Model::train(self : Model, num_steps : Int) -> TrainReport {
  if num_steps <= 0 {
    return { losses: [] }
  }

  let losses : Array[Double] = []
  for step = 0; step < num_steps; step = step + 1 {
    let doc = self.docs[step % self.docs.length()]
    let loss = self.loss_for_doc(doc)
    loss.backward()
    self.adam_update(step, num_steps)
    losses.push(loss.data)
  }
  { losses, }
}

///|
fn softmax_data(logits : Array[Value], temperature : Double) -> Array[Double] {
  if logits.length() == 0 {
    return []
  }

  let mut max_data = logits[0].data / temperature
  for i = 1; i < logits.length(); i = i + 1 {
    let scaled = logits[i].data / temperature
    if scaled > max_data {
      max_data = scaled
    }
  }

  let exps : Array[Double] = []
  let mut total = 0.0
  for logit in logits {
    let e = @math.exp(logit.data / temperature - max_data)
    exps.push(e)
    total = total + e
  }

  let probs : Array[Double] = []
  for e in exps {
    probs.push(e / total)
  }
  probs
}

///|
fn sample_from_probs(rand : @random.Rand, probs : Array[Double]) -> Int {
  let threshold = rand.double()
  let mut cum = 0.0
  for i = 0; i < probs.length(); i = i + 1 {
    cum = cum + probs[i]
    if threshold <= cum {
      return i
    }
  }
  probs.length() - 1
}

///|
fn Model::sample_one(self : Model, temperature : Double) -> String {
  let keys_cache : Array[Array[Array[Value]]] = Array::makei(
    self.config.n_layer,
    fn(_) { [] },
  )
  let values_cache : Array[Array[Array[Value]]] = Array::makei(
    self.config.n_layer,
    fn(_) { [] },
  )

  let mut token_id = self.tokenizer.bos
  let out : Array[Char] = []

  for pos_id = 0; pos_id < self.config.block_size; pos_id = pos_id + 1 {
    let logits = self.gpt_step(token_id, pos_id, keys_cache, values_cache)
    let probs = softmax_data(logits, temperature)
    token_id = sample_from_probs(self.rng, probs)
    if token_id == self.tokenizer.bos {
      break
    }
    out.push(self.tokenizer.chars[token_id])
  }

  String::from_array(out)
}

///|
pub fn Model::sample(
  self : Model,
  num_samples : Int,
  temperature : Double,
) -> Array[String] {
  if num_samples <= 0 {
    return []
  }
  let temp = if temperature <= 0.0 { 0.01 } else { temperature }

  let samples : Array[String] = []
  for i = 0; i < num_samples; i = i + 1 {
    samples.push(self.sample_one(temp))
  }
  samples
}
