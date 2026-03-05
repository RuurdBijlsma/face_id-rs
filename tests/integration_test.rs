use face_id::detector::{DetectorConfig, Face, ScrfdDetector};
use serde::Deserialize;
use std::fs::File;
use std::path::Path;

#[derive(Deserialize)]
struct ImageTestResult {
    filename: String,
    faces: Vec<Face>,
}

fn assert_approx_eq(actual: f32, expected: f32, label: &str) {
    assert!(
        (actual - expected).abs() < 1e-3,
        "{label}: {actual} != {expected}",
    );
}

#[test]
fn test_regression_against_json() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "assets/models/34g_gnkps.onnx";
    let json_path = "assets/reference_output/test_data.json";
    let img_dir = "assets/img";

    let file = File::open(json_path)?;
    let expected_results: Vec<ImageTestResult> = serde_json::from_reader(file)?;
    let mut detector = ScrfdDetector::new(model_path, DetectorConfig::default())?;

    for expected in expected_results {
        let img_path = Path::new(img_dir).join(&expected.filename);
        let img = image::open(img_path)?;
        let mut actual_faces = detector.detect(&img)?;

        assert_eq!(
            actual_faces.len(),
            expected.faces.len(),
            "Count mismatch for {}",
            expected.filename
        );

        actual_faces.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        let mut expected_faces = expected.faces;
        expected_faces.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        for (i, (actual, exp)) in actual_faces.iter().zip(expected_faces.iter()).enumerate() {
            let ctx = format!("{}:{}", expected.filename, i);
            // Compare Score
            assert_approx_eq(actual.score, exp.score, &format!("{ctx} Score", ));

            // Compare BBox
            assert_approx_eq(actual.bbox.x1, exp.bbox.x1, &format!("{ctx} BBox x1"));
            assert_approx_eq(actual.bbox.y1, exp.bbox.y1, &format!("{ctx} BBox y1"));
            assert_approx_eq(actual.bbox.x2, exp.bbox.x2, &format!("{ctx} BBox x2"));
            assert_approx_eq(actual.bbox.y2, exp.bbox.y2, &format!("{ctx} BBox y2"));

            // Compare Landmarks
            match (&actual.landmarks, &exp.landmarks) {
                (Some(a_lms), Some(e_lms)) => {
                    assert_eq!(a_lms.len(), e_lms.len(), "{ctx} Landmark count mismatch");
                    for (j, (a_pt, e_pt)) in a_lms.iter().zip(e_lms.iter()).enumerate() {
                        assert_approx_eq(a_pt.0, e_pt.0, &format!("{ctx} Lmk {j} x"));
                        assert_approx_eq(a_pt.1, e_pt.1, &format!("{ctx} Lmk {j} y"));
                    }
                }
                (None, None) => {}
                _ => panic!(
                    "{ctx} Landmark presence mismatch (One has landmarks, one doesn't)",
                ),
            }
        }
    }
    Ok(())
}
