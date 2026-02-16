///|
fn mean3(a : Double, b : Double, c : Double) -> Double {
  (a + b + c) / 3.0
}

///|
test "tokenizer: round trip" {
  let tokenizer = @microgpt.build_tokenizer(["emma", "noah"])
  let tokens = tokenizer.encode_doc("emma")

  assert_eq(tokenizer.decode_tokens(tokens), "emma")
  assert_eq(tokens[0], tokenizer.bos)
  assert_eq(tokens[tokens.length() - 1], tokenizer.bos)
}

///|
test "model: trains and loss trends down" {
  let docs = ["emma", "olivia", "ava", "liam", "noah", "oliver"]
  let model = @microgpt.Model::new(docs, @microgpt.default_config())
  let report = model.train(40)

  assert_eq(report.losses.length(), 40)

  let first = mean3(report.losses[0], report.losses[1], report.losses[2])
  let n = report.losses.length()
  let last = mean3(
    report.losses[n - 1],
    report.losses[n - 2],
    report.losses[n - 3],
  )
  inspect(last < first, content="true")
}

///|
test "model: sampling is deterministic" {
  let docs = ["emma", "olivia", "ava", "liam", "noah", "oliver"]
  let model = @microgpt.Model::new(docs, @microgpt.default_config())
  ignore(model.train(60))

  let samples = model.sample(5, 0.5)
  inspect(
    samples,
    content=(
      #|["liavia", "emav", "olia", "olivia", "ava"]
    ),
  )
}
