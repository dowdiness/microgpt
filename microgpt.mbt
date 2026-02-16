///|
/// Trains a microGPT model and returns generated samples.
///
/// This is a convenience helper around `Model::new`, `Model::train`, and
/// `Model::sample`.
pub fn run_microgpt(
  docs : Array[String],
  num_steps : Int,
  num_samples : Int,
  temperature : Double,
  config : Config,
) -> (Model, TrainReport, Array[String]) {
  let model = Model::new(docs, config)
  let report = model.train(num_steps)
  let samples = model.sample(num_samples, temperature)
  (model, report, samples)
}
