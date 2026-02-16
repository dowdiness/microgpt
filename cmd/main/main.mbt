///|
fn main {
  let docs = [
    "emma", "olivia", "ava", "isabella", "sophia", "mia", "charlotte", "amelia",
    "harper", "evelyn", "liam", "noah", "oliver", "elijah", "james", "william", "benjamin",
    "lucas", "henry", "theodore",
  ]

  let config = @microgpt.default_config()
  let (model, report, samples) = @microgpt.run_microgpt(
    docs, 200, 10, 0.5, config,
  )

  println("num params: " + model.num_params().to_string())
  if report.losses.length() > 0 {
    let first_loss = report.losses[0]
    let last_loss = report.losses[report.losses.length() - 1]
    println("loss: " + first_loss.to_string() + " -> " + last_loss.to_string())
  }

  for i = 0; i < samples.length(); i = i + 1 {
    println("sample " + (i + 1).to_string() + ": " + samples[i])
  }
}
