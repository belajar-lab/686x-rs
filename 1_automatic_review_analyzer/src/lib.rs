/// Finds the hinge loss on a single data point given specific classification
/// parameters.
/// Args:
///   - `feature_vector` - array describing the given data point.
///   - `label` - float, the correct classification of the data point.
///   - `theta` - array describing the linear classifier.
///   - `theta_0` - float representing the offset parameter.
/// Returns:
///   - the hinge loss, as a float, associated with the given data point and
///     parameters.
pub fn hinge_loss_single(feature_vector: &[f64], label: f64, theta: &[f64], theta_0: f64) -> f64 {
    let y = feature_vector.iter()
        .zip(theta.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>() + theta_0;

    (1.0 - y * label).max(0.0)
}