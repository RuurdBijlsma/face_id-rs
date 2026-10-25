#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]

use color_eyre::eyre::{Result, bail};
use comfy_table::{Cell, Color, ContentArrangement, Table, presets::UTF8_FULL};
use face_id::analyzer::{FaceAnalysis, FaceAnalyzer};
use image::GenericImageView;
use std::fs;
use std::path::Path;

#[derive(serde::Deserialize)]
struct ReferenceEntry {
    filename: String,
    results: Vec<FaceAnalysis>,
}

#[derive(Default)]
struct MetricTracker {
    values: Vec<f64>,
}

impl MetricTracker {
    fn add(&mut self, val: f64) {
        self.values.push(val);
    }

    fn min(&self) -> f64 {
        self.values.iter().copied().fold(f64::INFINITY, f64::min)
    }

    fn max(&self) -> f64 {
        self.values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    fn mean(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.values.iter().sum::<f64>() / self.values.len() as f64
        }
    }

    const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[derive(Default)]
struct ConsistencyReport {
    total_ref_faces: usize,
    total_live_faces: usize,
    matched_faces: usize,
    unmatched_ref_faces: usize,
    unmatched_live_faces: usize,

    gender_matches: usize,
    age_diffs: MetricTracker,
    score_diffs: MetricTracker,
    bbox_ious: MetricTracker,
    landmark_drift_px: MetricTracker,
    emb_l_inf_diffs: MetricTracker,
    emb_cosine_dists: MetricTracker,
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let img_dir = Path::new("assets/img");
    let reference_path = Path::new("assets/reference_analysis.json");

    if !reference_path.exists() {
        bail!(
            "Reference file '{}' not found. Run 'cargo run --example update_reference' first.",
            reference_path.display()
        );
    }

    println!("Loading reference data from: {}", reference_path.display());
    let ref_file = fs::File::open(reference_path)?;
    let ref_entries: Vec<ReferenceEntry> = serde_json::from_reader(ref_file)?;

    println!("Initializing FaceAnalyzer...");
    let analyzer = FaceAnalyzer::from_hf().build().await?;

    println!(
        "\nEvaluating consistency across {} images...",
        ref_entries.len()
    );
    let mut report = ConsistencyReport::default();

    for entry in &ref_entries {
        let img_path = img_dir.join(&entry.filename);
        if !img_path.exists() {
            eprintln!(
                "Warning: Image file '{}' not found, skipping.",
                img_path.display()
            );
            continue;
        }

        let img = image::open(&img_path)?;
        let (width, height) = img.dimensions();
        let live_results = analyzer.analyze(&img)?;

        let num_ref = entry.results.len();
        let num_live = live_results.len();
        report.total_ref_faces += num_ref;
        report.total_live_faces += num_live;

        // Perform greedy IoU matching between reference faces and live detections
        let (matches, unmatched_ref, unmatched_live) =
            match_faces(&entry.results, &live_results, 0.5);

        report.matched_faces += matches.len();
        report.unmatched_ref_faces += unmatched_ref.len();
        report.unmatched_live_faces += unmatched_live.len();

        let mut img_max_emb_diff = 0.0f32;
        let mut img_max_cosine_dist = 0.0f32;
        let mut img_min_iou = 1.0f32;

        for &(ref_idx, live_idx) in &matches {
            let ref_face = &entry.results[ref_idx];
            let live_face = &live_results[live_idx];

            // 1. Detection Score Difference
            let score_diff = (live_face.detection.score - ref_face.detection.score).abs();
            report.score_diffs.add(f64::from(score_diff));

            // 2. Bounding Box IoU
            let iou = live_face.detection.bbox.iou(&ref_face.detection.bbox);
            report.bbox_ious.add(f64::from(iou));
            img_min_iou = img_min_iou.min(iou);

            // 3. Landmarks Pixel Drift (Euclidean distance on 5 keypoints)
            if let (Some(live_lms), Some(ref_lms)) = (
                &live_face.detection.landmarks,
                &ref_face.detection.landmarks,
            ) {
                for (pt_live, pt_ref) in live_lms.iter().zip(ref_lms.iter()) {
                    let dx = (pt_live.0 - pt_ref.0) * width as f32;
                    let dy = (pt_live.1 - pt_ref.1) * height as f32;
                    let dist_px = dx.hypot(dy);
                    report.landmark_drift_px.add(f64::from(dist_px));
                }
            }

            // 4. Gender Agreement
            if live_face.gender == ref_face.gender {
                report.gender_matches += 1;
            }

            // 5. Age Absolute Difference
            let age_diff = (i32::from(live_face.age) - i32::from(ref_face.age)).abs();
            report.age_diffs.add(f64::from(age_diff));

            // 6. Embedding Metrics
            let (cosine_dist, l_inf_diff) =
                compute_embedding_metrics(&live_face.embedding, &ref_face.embedding);

            report.emb_cosine_dists.add(f64::from(cosine_dist));
            report.emb_l_inf_diffs.add(f64::from(l_inf_diff));

            img_max_emb_diff = img_max_emb_diff.max(l_inf_diff);
            img_max_cosine_dist = img_max_cosine_dist.max(cosine_dist);
        }

        println!(
            "  - {:<18} | Ref: {} | Live: {} | Matched: {} | Max L∞ Emb: {:.6} | Cosine Dist: {:.6} | Min IoU: {:.4}",
            entry.filename,
            num_ref,
            num_live,
            matches.len(),
            img_max_emb_diff,
            img_max_cosine_dist,
            if matches.is_empty() { 0.0 } else { img_min_iou }
        );

        if !unmatched_ref.is_empty() {
            eprintln!(
                "    ⚠️  Missing {} reference face(s) in {}",
                unmatched_ref.len(),
                entry.filename
            );
        }
        if !unmatched_live.is_empty() {
            eprintln!(
                "    ⚠️  Found {} unexpected live face(s) in {}",
                unmatched_live.len(),
                entry.filename
            );
        }
    }

    // Render Summary Table
    let all_passed = display_summary_table(&report);

    if all_passed {
        println!("\n✅ Consistency check PASSED: All generated values are within tolerance.\n");
        Ok(())
    } else {
        bail!("Consistency check FAILED: One or more metrics exceeded expected tolerance.");
    }
}

/// Matches reference faces to live faces using greedy bipartite `IoU` matching.
fn match_faces(
    ref_faces: &[FaceAnalysis],
    live_faces: &[FaceAnalysis],
    iou_threshold: f32,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let mut candidate_pairs = Vec::new();
    for (r_idx, r_face) in ref_faces.iter().enumerate() {
        for (l_idx, l_face) in live_faces.iter().enumerate() {
            let iou = r_face.detection.bbox.iou(&l_face.detection.bbox);
            if iou >= iou_threshold {
                candidate_pairs.push((r_idx, l_idx, iou));
            }
        }
    }

    // Sort descending by IoU
    candidate_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut matches = Vec::new();
    let mut matched_live = vec![false; live_faces.len()];
    let mut matched_ref = vec![false; ref_faces.len()];

    for (r_idx, l_idx, _) in candidate_pairs {
        if !matched_ref[r_idx] && !matched_live[l_idx] {
            matched_ref[r_idx] = true;
            matched_live[l_idx] = true;
            matches.push((r_idx, l_idx));
        }
    }

    let unmatched_ref: Vec<usize> = matched_ref
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { None } else { Some(i) })
        .collect();

    let unmatched_live: Vec<usize> = matched_live
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { None } else { Some(i) })
        .collect();

    (matches, unmatched_ref, unmatched_live)
}

/// Computes Cosine Distance (1 - cos) and `L_infinity` norm (max absolute element difference).
fn compute_embedding_metrics(emb1: &[f32], emb2: &[f32]) -> (f32, f32) {
    let mut dot_product = 0.0f32;
    let mut max_abs_diff = 0.0f32;

    for (&a, &b) in emb1.iter().zip(emb2.iter()) {
        dot_product = f32::mul_add(a, b, dot_product);
        let diff = (a - b).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }

    let cosine_dist = (1.0 - dot_product).max(0.0);
    (cosine_dist, max_abs_diff)
}

/// Formats and prints the overall statistical summary table. Returns true if all metrics pass.
fn display_summary_table(report: &ConsistencyReport) -> bool {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Metric", "Min", "Mean", "Max", "Tolerance", "Status"]);

    let mut all_passed = true;

    // 1. Face Count Match
    let count_match = report.matched_faces == report.total_ref_faces
        && report.unmatched_live_faces == 0
        && report.unmatched_ref_faces == 0;
    all_passed &= count_match;
    table.add_row(vec![
        Cell::new("Face Count Match"),
        Cell::new("-"),
        Cell::new(format!(
            "{}/{} faces",
            report.matched_faces, report.total_ref_faces
        )),
        Cell::new(format!(
            "Missing: {}, Extra: {}",
            report.unmatched_ref_faces, report.unmatched_live_faces
        )),
        Cell::new("Exact (0 diff)"),
        status_cell(count_match),
    ]);

    // 2. Gender Match
    let gender_pct = if report.matched_faces > 0 {
        (report.gender_matches as f64 / report.matched_faces as f64) * 100.0
    } else {
        100.0
    };
    let gender_pass = (gender_pct - 100.0).abs() < f64::EPSILON;
    all_passed &= gender_pass;
    table.add_row(vec![
        Cell::new("Gender Match Rate"),
        Cell::new("-"),
        Cell::new(format!("{gender_pct:.1}%")),
        Cell::new("-"),
        Cell::new("100%"),
        status_cell(gender_pass),
    ]);

    // 3. Age Diff
    let age_pass = !report.age_diffs.is_empty() && report.age_diffs.max() == 0.0;
    all_passed &= age_pass;
    table.add_row(vec![
        Cell::new("Age Absolute Error"),
        Cell::new(format!("{:.0} yrs", report.age_diffs.min())),
        Cell::new(format!("{:.2} yrs", report.age_diffs.mean())),
        Cell::new(format!("{:.0} yrs", report.age_diffs.max())),
        Cell::new("0 yrs"),
        status_cell(age_pass),
    ]);

    // 4. Detection Score
    let score_tol = 1e-3;
    let score_pass = !report.score_diffs.is_empty() && report.score_diffs.max() <= score_tol;
    all_passed &= score_pass;
    table.add_row(vec![
        Cell::new("Detection Score |Δ|"),
        Cell::new(format!("{:.6}", report.score_diffs.min())),
        Cell::new(format!("{:.6}", report.score_diffs.mean())),
        Cell::new(format!("{:.6}", report.score_diffs.max())),
        Cell::new("≤ 0.001"),
        status_cell(score_pass),
    ]);

    // 5. Bounding Box IoU
    let iou_tol = 0.99;
    let iou_pass = !report.bbox_ious.is_empty() && report.bbox_ious.min() >= iou_tol;
    all_passed &= iou_pass;
    table.add_row(vec![
        Cell::new("Bounding Box IoU"),
        Cell::new(format!("{:.4}", report.bbox_ious.min())),
        Cell::new(format!("{:.4}", report.bbox_ious.mean())),
        Cell::new(format!("{:.4}", report.bbox_ious.max())),
        Cell::new("≥ 0.99"),
        status_cell(iou_pass),
    ]);

    // 6. Landmark Drift
    let landmark_tol_px = 1.0;
    let landmark_pass =
        report.landmark_drift_px.is_empty() || report.landmark_drift_px.max() <= landmark_tol_px;
    all_passed &= landmark_pass;
    table.add_row(vec![
        Cell::new("Landmark Drift (px)"),
        Cell::new(format!("{:.3} px", report.landmark_drift_px.min())),
        Cell::new(format!("{:.3} px", report.landmark_drift_px.mean())),
        Cell::new(format!("{:.3} px", report.landmark_drift_px.max())),
        Cell::new("≤ 1.0 px"),
        status_cell(landmark_pass),
    ]);

    // 7. Embedding L_inf (Max Dimension Diff)
    let emb_tol = 1e-3;
    let emb_pass = !report.emb_l_inf_diffs.is_empty() && report.emb_l_inf_diffs.max() <= emb_tol;
    all_passed &= emb_pass;
    table.add_row(vec![
        Cell::new("Embedding Max Diff (L∞)"),
        Cell::new(format!("{:.6}", report.emb_l_inf_diffs.min())),
        Cell::new(format!("{:.6}", report.emb_l_inf_diffs.mean())),
        Cell::new(format!("{:.6}", report.emb_l_inf_diffs.max())),
        Cell::new("≤ 0.001"),
        status_cell(emb_pass),
    ]);

    // 8. Embedding Cosine Distance
    let cosine_tol = 1e-3;
    let cosine_pass =
        !report.emb_cosine_dists.is_empty() && report.emb_cosine_dists.max() <= cosine_tol;
    all_passed &= cosine_pass;
    table.add_row(vec![
        Cell::new("Embedding Cosine Dist"),
        Cell::new(format!("{:.6}", report.emb_cosine_dists.min())),
        Cell::new(format!("{:.6}", report.emb_cosine_dists.mean())),
        Cell::new(format!("{:.6}", report.emb_cosine_dists.max())),
        Cell::new("≤ 0.001"),
        status_cell(cosine_pass),
    ]);

    println!("\n{table}");
    all_passed
}

fn status_cell(passed: bool) -> Cell {
    if passed {
        Cell::new("PASS").fg(Color::Green)
    } else {
        Cell::new("FAIL").fg(Color::Red)
    }
}
