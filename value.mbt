///|
pub(open) trait AutogradEngine {
  zero() -> Self
  scalar(Double) -> Self
  pow(Self, Double) -> Self
  log(Self) -> Self
  exp(Self) -> Self
  relu(Self) -> Self
  backward(Self) -> Unit
}

///|
priv struct Edge {
  child : Value
  local_grad : Double
}

///|
struct Value {
  id : Int
  mut data : Double
  mut grad : Double
  children : Array[Edge]
}

///|
let next_value_id : Ref[Int] = { val: 0 }

///|
fn fresh_value_id() -> Int {
  let id = next_value_id.val
  next_value_id.val = id + 1
  id
}

///|
fn new_value(data : Double, children : Array[Edge]) -> Value {
  { id: fresh_value_id(), data, grad: 0.0, children }
}

///|
fn value(data : Double) -> Value {
  new_value(data, [])
}

///|
impl AutogradEngine for Value with zero() -> Value {
  new_value(0.0, [])
}

///|
impl AutogradEngine for Value with scalar(data : Double) -> Value {
  new_value(data, [])
}

///|
impl Add for Value with add(self : Value, other : Value) -> Value {
  new_value(self.data + other.data, [
    { child: self, local_grad: 1.0 },
    { child: other, local_grad: 1.0 },
  ])
}

///|
impl Mul for Value with mul(self : Value, other : Value) -> Value {
  new_value(self.data * other.data, [
    { child: self, local_grad: other.data },
    { child: other, local_grad: self.data },
  ])
}

///|
impl AutogradEngine for Value with pow(self : Value, exponent : Double) -> Value {
  let pow_data = @math.pow(self.data, exponent)
  new_value(pow_data, [
    { child: self, local_grad: exponent * @math.pow(self.data, exponent - 1.0) },
  ])
}

///|
impl AutogradEngine for Value with log(self : Value) -> Value {
  new_value(@math.ln(self.data), [{ child: self, local_grad: 1.0 / self.data }])
}

///|
impl AutogradEngine for Value with exp(self : Value) -> Value {
  let exp_data = @math.exp(self.data)
  new_value(exp_data, [{ child: self, local_grad: exp_data }])
}

///|
impl AutogradEngine for Value with relu(self : Value) -> Value {
  let relu_data = if self.data > 0.0 { self.data } else { 0.0 }
  let local_grad = if self.data > 0.0 { 1.0 } else { 0.0 }
  new_value(relu_data, [{ child: self, local_grad }])
}

///|
impl Neg for Value with neg(self : Value) -> Value {
  self * Value::scalar(-1.0)
}

///|
impl Sub for Value with sub(self : Value, other : Value) -> Value {
  self + -other
}

///|
impl Div for Value with div(self : Value, other : Value) -> Value {
  self * other.pow(-1.0)
}

///|
fn build_topological_order(
  v : Value,
  visited : @hashset.HashSet[Int],
  topo : Array[Value],
) -> Unit {
  if visited.contains(v.id) {
    return
  }
  visited.add(v.id)
  for edge in v.children {
    build_topological_order(edge.child, visited, topo)
  }
  topo.push(v)
}

///|
impl AutogradEngine for Value with backward(self : Value) -> Unit {
  let topo : Array[Value] = []
  let visited : @hashset.HashSet[Int] = @hashset.new()
  build_topological_order(self, visited, topo)
  self.grad = 1.0
  for i = topo.length() - 1; i >= 0; i = i - 1 {
    let node = topo[i]
    for edge in node.children {
      edge.child.grad = edge.child.grad + edge.local_grad * node.grad
    }
  }
}

///|
fn[T : AutogradEngine + Add] sum_values(xs : Array[T]) -> T {
  let mut out = T::zero()
  for x in xs {
    out = out + x
  }
  out
}
