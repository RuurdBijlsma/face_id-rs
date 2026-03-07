use color_eyre::eyre::Result;
use face_id::detector::{DetectedFace, ScrfdDetector};
use serde::Serialize;
use std::fs::{self, File};
use std::path::Path;

#[derive(Serialize)]
struct ImageTestResult {
    filename: String,
    faces: Vec<DetectedFace>,
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let model_path = "assets/models/34g_gnkps.onnx";
    let img_dir = "assets/img";
    let output_json = "assets/reference_output/test_data.json";

    // Ensure the output directory exists
    if let Some(parent) = Path::new(output_json).parent() {
        fs::create_dir_all(parent)?;
    }

    let mut detector = ScrfdDetector::builder(model_path).build()?;
    let mut all_results = Vec::new();

    let entries = fs::read_dir(img_dir)?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        // Filter for image files
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

        // 1. Load image using the 'image' crate
        let img = image::open(&path)?;

        // 2. Run detection
        let faces = detector.detect(&img)?;
        println!("  -> Found {} faces", faces.len());

        all_results.push(ImageTestResult { filename, faces });
    }

    // 3. Write results to JSON file
    let file = File::create(output_json)?;
    serde_json::to_writer_pretty(file, &all_results)?;

    println!("\nSuccess! Results written to {}", output_json);
    Ok(())
}
