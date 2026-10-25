#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]

use face_id::analyzer::{FaceAnalysis, FaceAnalyzer};
use face_id::embedder::ArcFaceEmbedder;
use std::cmp::Ordering;
use std::fs;
use std::path::Path;

const EPSILON: f32 = 1e-3;

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON
}

#[derive(serde::Deserialize)]
struct ReferenceEntry {
    filename: String,
    results: Vec<FaceAnalysis>,
}

#[tokio::test]
async fn test_analyzer_consistency_with_reference() -> color_eyre::Result<()> {
    let _ = color_eyre::install();
    let img_dir = "assets/img";
    let reference_path = "assets/reference_analysis.json";

    assert!(
        Path::new(reference_path).exists(),
        "Reference file {reference_path} not found. Run 'cargo run --example update_reference' first."
    );

    let ref_file = fs::File::open(reference_path)?;
    let reference_list: Vec<ReferenceEntry> = serde_json::from_reader(ref_file)?;
    let analyzer = FaceAnalyzer::from_hf().build().await?;

    for ref_entry in reference_list {
        let filename = &ref_entry.filename;
        let ref_results = &ref_entry.results;

        let img_path = Path::new(img_dir).join(filename);
        let img = image::open(&img_path).unwrap_or_else(|_| panic!("Failed to open {filename}"));

        // Run live analysis
        let live_results = analyzer.analyze(&img)?;

        // 1. Verify Face Count
        assert_eq!(
            live_results.len(),
            ref_results.len(),
            "Face count mismatch for image: {filename}. Expected {}, got {}",
            ref_results.len(),
            live_results.len()
        );

        // 2. IoU for face pairs
        let mut candidate_pairs = Vec::new();
        for (ref_idx, ref_face) in ref_results.iter().enumerate() {
            for (live_idx, live_face) in live_results.iter().enumerate() {
                let iou = ref_face.detection.bbox.iou(&live_face.detection.bbox);
                if iou >= 0.5 {
                    candidate_pairs.push((ref_idx, live_idx, iou));
                }
            }
        }

        // Sort descending by IoU so strongest matches are claimed first
        candidate_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));

        let mut matched_ref = vec![false; ref_results.len()];
        let mut matched_live = vec![false; live_results.len()];
        let mut matches = Vec::with_capacity(ref_results.len());

        for (r_idx, l_idx, _) in candidate_pairs {
            if !matched_ref[r_idx] && !matched_live[l_idx] {
                matched_ref[r_idx] = true;
                matched_live[l_idx] = true;
                matches.push((r_idx, l_idx));
            }
        }

        assert_eq!(
            matches.len(),
            ref_results.len(),
            "Not all faces could be matched (IoU >= 0.5) in {filename}. Matched {}/{}",
            matches.len(),
            ref_results.len()
        );

        for &(ref_idx, live_idx) in &matches {
            let ref_face = &ref_results[ref_idx];
            let live_face = &live_results[live_idx];

            // 3. Check Detection Score
            assert!(
                approx_eq(live_face.detection.score, ref_face.detection.score),
                "Score mismatch in {filename} for face {ref_idx}: live={}, ref={}",
                live_face.detection.score,
                ref_face.detection.score
            );

            // 4. Check Bounding Box Coordinates
            let (lb, rb) = (&live_face.detection.bbox, &ref_face.detection.bbox);
            assert!(
                approx_eq(lb.x1, rb.x1)
                    && approx_eq(lb.y1, rb.y1)
                    && approx_eq(lb.x2, rb.x2)
                    && approx_eq(lb.y2, rb.y2),
                "BBox mismatch in {filename} for face {ref_idx}:\n  live: [{:.4}, {:.4}, {:.4}, {:.4}]\n  ref:  [{:.4}, {:.4}, {:.4}, {:.4}]",
                lb.x1,
                lb.y1,
                lb.x2,
                lb.y2,
                rb.x1,
                rb.y1,
                rb.x2,
                rb.y2
            );

            // 5. Check Landmarks (if present)
            if let (Some(live_lms), Some(ref_lms)) = (
                &live_face.detection.landmarks,
                &ref_face.detection.landmarks,
            ) {
                assert_eq!(live_lms.len(), ref_lms.len());
                for (j, (pt_live, pt_ref)) in live_lms.iter().zip(ref_lms.iter()).enumerate() {
                    assert!(
                        approx_eq(pt_live.0, pt_ref.0) && approx_eq(pt_live.1, pt_ref.1),
                        "Landmark {j} mismatch in {filename} for face {ref_idx}: live=({:.4}, {:.4}), ref=({:.4}, {:.4})",
                        pt_live.0,
                        pt_live.1,
                        pt_ref.0,
                        pt_ref.1
                    );
                }
            }

            // 6. Check Gender & Age
            assert_eq!(
                live_face.gender, ref_face.gender,
                "Gender mismatch in {filename} for face {ref_idx}: live={:?}, ref={:?}",
                live_face.gender, ref_face.gender
            );
            assert_eq!(
                live_face.age, ref_face.age,
                "Age mismatch in {filename} for face {ref_idx}: live={}, ref={}",
                live_face.age, ref_face.age
            );

            // 7. Check Embedding Consistency
            assert_eq!(
                live_face.embedding.len(),
                ref_face.embedding.len(),
                "Embedding length mismatch in {filename}"
            );

            let cosine_sim =
                ArcFaceEmbedder::compute_similarity(&live_face.embedding, &ref_face.embedding);
            let cosine_dist = (1.0 - cosine_sim).max(0.0);
            assert!(
                cosine_dist < EPSILON,
                "Embedding cosine distance mismatch in {filename} for face {ref_idx}: dist={cosine_dist:.6} (expected < {EPSILON})"
            );

            for (dim, (&live_val, &ref_val)) in live_face
                .embedding
                .iter()
                .zip(ref_face.embedding.iter())
                .enumerate()
            {
                assert!(
                    approx_eq(live_val, ref_val),
                    "Embedding dim {dim} mismatch in {filename} for face {ref_idx}: live={live_val}, ref={ref_val}, diff={}",
                    (live_val - ref_val).abs()
                );
            }
        }
        println!("Verified consistency for {filename}");
    }

    Ok(())
}
