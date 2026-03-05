use detect_faces_test::detector::{DetectorConfig, Face, ScrfdDetector};
use opencv::core::MatTraitConst;
use opencv::imgcodecs;
use serde::Deserialize;
use std::fs::File;
use std::path::Path;

// Tolerance for floating point comparisons
const EPSILON: f32 = 1e-3;

#[derive(Deserialize)]
struct ImageTestResult {
    filename: String,
    faces: Vec<Face>,
}

fn assert_approx_eq(actual: f32, expected: f32, label: &str) {
    assert!(
        (actual - expected).abs() < EPSILON,
        "{}: {} is not approx {}, diff: {}",
        label,
        actual,
        expected,
        (actual - expected).abs()
    );
}

#[test]
fn test_regression_against_json() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "assets/models/34g_gnkps.onnx";
    let json_path = "assets/reference_output/test_data.json";
    let img_dir = "assets/img";

    // 1. Load the expected results from JSON
    let file = File::open(json_path).map_err(|e| {
        format!(
            "Could not open {}. Did you run 'cargo run --example produce_test_json'? Error: {}",
            json_path, e
        )
    })?;
    let expected_results: Vec<ImageTestResult> = serde_json::from_reader(file)?;

    // 2. Initialize detector
    let mut detector = ScrfdDetector::new(model_path, DetectorConfig::default())?;

    for expected in expected_results {
        let img_path = Path::new(img_dir).join(&expected.filename);
        println!("Testing regression for: {}", expected.filename);

        // 3. Run current detection logic
        let img = imgcodecs::imread(img_path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
        assert!(!img.empty(), "Failed to load image: {}", expected.filename);

        let mut actual_faces = detector.detect(&img)?;

        // 4. Compare Face Counts
        assert_eq!(
            actual_faces.len(),
            expected.faces.len(),
            "Face count changed for {}",
            expected.filename
        );

        // 5. Sort both by score to ensure deterministic comparison
        actual_faces.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        let mut expected_faces = expected.faces;
        expected_faces.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // 6. Detailed Comparison
        for (i, (actual, exp)) in actual_faces.iter().zip(expected_faces.iter()).enumerate() {
            let context = format!("Img: {}, Face: {}", expected.filename, i);

            // Compare Score
            assert_approx_eq(actual.score, exp.score, &format!("{} Score", context));

            // Compare BBox
            assert_approx_eq(actual.bbox.x1, exp.bbox.x1, &format!("{} BBox x1", context));
            assert_approx_eq(actual.bbox.y1, exp.bbox.y1, &format!("{} BBox y1", context));
            assert_approx_eq(actual.bbox.x2, exp.bbox.x2, &format!("{} BBox x2", context));
            assert_approx_eq(actual.bbox.y2, exp.bbox.y2, &format!("{} BBox y2", context));

            // Compare Landmarks
            match (&actual.landmarks, &exp.landmarks) {
                (Some(a_lms), Some(e_lms)) => {
                    assert_eq!(
                        a_lms.len(),
                        e_lms.len(),
                        "{} Landmark count mismatch",
                        context
                    );
                    for (j, (a_pt, e_pt)) in a_lms.iter().zip(e_lms.iter()).enumerate() {
                        assert_approx_eq(a_pt.0, e_pt.0, &format!("{} Lmk {} x", context, j));
                        assert_approx_eq(a_pt.1, e_pt.1, &format!("{} Lmk {} y", context, j));
                    }
                }
                (None, None) => {}
                _ => panic!(
                    "{} Landmark presence mismatch (One has landmarks, one doesn't)",
                    context
                ),
            }
        }
    }

    Ok(())
}
