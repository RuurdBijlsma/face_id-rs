use color_eyre::eyre::Result;
use detect_faces_test::detector::{DetectorConfig, Face, ScrfdDetector};
use opencv::core::MatTraitConst;
use opencv::imgcodecs;
use serde::Serialize;
use std::fs::File;

#[derive(Serialize)]
struct ImageTestResult {
    filename: String,
    faces: Vec<Face>,
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let model_path = "assets/models/34g_gnkps.onnx";
    let img_dir = "assets/img";
    let output_json = "assets/reference_output/test_data.json";

    let mut detector = ScrfdDetector::new(model_path, DetectorConfig::default())?;
    let mut all_results = Vec::new();

    let entries = std::fs::read_dir(img_dir)?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        // Check if it's an image
        let extension = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        if !["jpg", "jpeg", "png"].contains(&extension.as_str()) {
            continue;
        }

        let filename = path.file_name().unwrap().to_string_lossy().into_owned();
        println!("Processing {}...", filename);

        // Load image
        let img = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
        if img.empty() {
            eprintln!("Could not read image: {}", filename);
            continue;
        }

        // Run detection
        let faces = detector.detect(&img)?;

        all_results.push(ImageTestResult { filename, faces });
    }

    // Write to JSON file
    let file = File::create(output_json)?;
    serde_json::to_writer_pretty(file, &all_results)?;

    println!("\nSuccess! Results written to {}", output_json);
    Ok(())
}
