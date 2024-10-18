use automatic_review_analyzer::hinge_loss_single;

#[test]
fn hinge_loss_single_equal_to_one() {
    let feature_vector = vec![
        0.34321631, 0.42850303, 0.06123605, 0.72269115, 0.34960656, 0.44380283, 0.25489608, 0.69820867, 0.83814653, 0.1348976,
        ];
    let label: f64 = 1.0;
    let theta = vec![
        0.14568072, 0.11668529, 0.81651253, 0.06918585, 0.14301791, 0.11266264, 0.19615837, 0.07161183, 0.05965544, 0.37065151,
    ];
    let theta_0: f64 = 0.5;
    let exp_result: f64 = 0.0;
    assert_eq!(hinge_loss_single(&feature_vector, label, &theta, theta_0), exp_result);
}