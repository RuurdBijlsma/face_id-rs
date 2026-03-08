#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use face_id::analyzer::FaceAnalyzer;
use ort::ep::CUDA;
use std::fs;
use std::path::Path;

const EPSILON: f32 = 1e-3;

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON
}

#[tokio::test]
async fn test_analyzer_consistency_with_reference() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let img_dir = "assets/img";
    let reference_path = "assets/reference_analysis.json";
    assert!(
        Path::new(reference_path).exists(),
        "Reference file {reference_path} not found. Run the comprehensive_analysis example first."
    );
    let ref_file = fs::File::open(reference_path)?;
    let reference_data: serde_json::Value = serde_json::from_reader(ref_file)?;
    let reference_list = reference_data.as_array().expect("JSON should be an array");
    let analyzer = FaceAnalyzer::from_hf()
        .with_execution_providers(&[CUDA::default().build().error_on_failure()])
        .build()
        .await?;

    for ref_entry in reference_list {
        let filename = ref_entry["filename"].as_str().unwrap();
        let ref_results = ref_entry["results"].as_array().unwrap();

        let img_path = Path::new(img_dir).join(filename);
        let img = image::open(&img_path).unwrap_or_else(|_| panic!("Failed to open {filename}"));

        // Run live analysis
        let live_results = analyzer.analyze(&img)?;

        // Check if face count matches
        assert_eq!(
            live_results.len(),
            ref_results.len(),
            "Face count mismatch for image: {filename}"
        );

        for (i, live_face) in live_results.iter().enumerate() {
            let ref_face = &ref_results[i];

            // Check Detection Score
            let ref_score = ref_face["detection"]["score"].as_f64().unwrap() as f32;
            assert!(
                approx_eq(live_face.detection.score, ref_score),
                "Score mismatch in {filename} for face {i}"
            );

            // Check Bounding Box
            let ref_bbox = &ref_face["detection"]["bbox"];
            assert!(approx_eq(
                live_face.detection.bbox.x1,
                ref_bbox["x1"].as_f64().unwrap() as f32
            ));
            assert!(approx_eq(
                live_face.detection.bbox.y1,
                ref_bbox["y1"].as_f64().unwrap() as f32
            ));
            assert!(approx_eq(
                live_face.detection.bbox.x2,
                ref_bbox["x2"].as_f64().unwrap() as f32
            ));
            assert!(approx_eq(
                live_face.detection.bbox.y2,
                ref_bbox["y2"].as_f64().unwrap() as f32
            ));

            // Check Landmarks (if present)
            if let Some(live_lms) = &live_face.detection.landmarks {
                let ref_lms = ref_face["detection"]["landmarks"].as_array().unwrap();
                for (j, pt) in live_lms.iter().enumerate() {
                    let ref_pt = &ref_lms[j];
                    assert!(approx_eq(pt.0, ref_pt[0].as_f64().unwrap() as f32));
                    assert!(approx_eq(pt.1, ref_pt[1].as_f64().unwrap() as f32));
                }
            }

            // Check Gender & Age
            if let Some(live_ga) = &live_face.gender_age {
                let ref_ga = &ref_face["gender_age"];

                // Compare Gender (Enum vs String/Int)
                let ref_gender_str = ref_ga["gender"].as_str().unwrap();
                let live_gender_str = format!("{:?}", live_ga.gender);
                assert_eq!(live_gender_str, ref_gender_str);

                // Compare Age
                let ref_age = ref_ga["age"].as_u64().unwrap() as u8;
                assert_eq!(live_ga.age, ref_age);
            }

            // Check Embedding Consistency
            if let Some(live_emb) = &live_face.embedding {
                let ref_emb = ref_face["embedding"].as_array().unwrap();
                assert_eq!(live_emb.len(), ref_emb.len());

                // Check every dimension (usually 512)
                for (dim, val) in live_emb.iter().enumerate() {
                    let ref_val = ref_emb[dim].as_f64().unwrap() as f32;
                    assert!(
                        approx_eq(*val, ref_val),
                        "Embedding dimension {dim} mismatch in {filename} for face {i}"
                    );
                }
            }
        }
        println!("Verified consistency for {filename}");
    }

    Ok(())
}
