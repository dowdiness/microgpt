///|
test "autograd: linear expression gradients" {
  let x = value(2.0)
  let y = value(-3.0)
  let z = value(10.0)
  let out = x * y + z

  out.backward()

  assert_eq(x.grad, -3.0)
  assert_eq(y.grad, 2.0)
  assert_eq(z.grad, 1.0)
}

///|
test "autograd: square gradient" {
  let x = value(3.0)
  let out = x * x

  out.backward()

  assert_eq(x.grad, 6.0)
}
