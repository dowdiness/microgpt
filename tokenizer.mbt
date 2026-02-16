///|
pub struct Tokenizer {
  chars : Array[Char]
  bos : Int
  stoi : @hashmap.HashMap[Char, Int]
}

///|
fn normalize_docs(docs : Array[String]) -> Array[String] {
  let out : Array[String] = []
  for doc in docs {
    let trimmed = doc.trim().to_string()
    if !trimmed.is_empty() {
      out.push(trimmed)
    }
  }
  out
}

///|
pub fn build_tokenizer(docs : Array[String]) -> Tokenizer {
  let clean_docs = normalize_docs(docs)
  if clean_docs.is_empty() {
    abort("build_tokenizer: docs must be non-empty")
  }

  let uniq : @hashset.HashSet[Char] = @hashset.new()
  for doc in clean_docs {
    for ch in doc.to_array() {
      uniq.add(ch)
    }
  }

  let chars = uniq.to_array()
  chars.sort()

  let stoi : @hashmap.HashMap[Char, Int] = @hashmap.new()
  for i = 0; i < chars.length(); i = i + 1 {
    stoi[chars[i]] = i
  }

  { chars, bos: chars.length(), stoi }
}

///|
pub fn Tokenizer::vocab_size(self : Tokenizer) -> Int {
  self.chars.length() + 1
}

///|
pub fn Tokenizer::encode_doc(self : Tokenizer, doc : String) -> Array[Int] {
  let tokens : Array[Int] = [self.bos]
  for ch in doc.to_array() {
    match self.stoi.get(ch) {
      Some(id) => tokens.push(id)
      None => abort("Tokenizer::encode_doc: unknown token")
    }
  }
  tokens.push(self.bos)
  tokens
}

///|
pub fn Tokenizer::decode_tokens(
  self : Tokenizer,
  token_ids : Array[Int],
) -> String {
  let out : Array[Char] = []
  for token_id in token_ids {
    if token_id == self.bos {
      continue
    }
    if token_id < 0 || token_id >= self.chars.length() {
      abort("Tokenizer::decode_tokens: token id out of range")
    }
    out.push(self.chars[token_id])
  }
  String::from_array(out)
}
