use color_eyre::eyre::Result;
use face_id::detector::ScrfdDetector;
use face_id::face_align::norm_crop;
use std::fs;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let model_id = "RuteNL/SCRFD-face-detection-ONNX";
    // Note: We must use a model that supports keypoints (kps) for alignment.
    let model_filename = "34g_gnkps.onnx";
    let img_dir = "assets/img";
    let output_base = "output_previews/aligned_faces";

    // 1. Initialize detector
    let mut detector = ScrfdDetector::from_hf(model_id, model_filename)
        .build()
        .await?;

    // 2. Prepare output directory
    if !Path::new(output_base).exists() {
        fs::create_dir_all(output_base)?;
    }

    println!("Using model: {}", model_filename);

    // 3. Process images
    for entry in fs::read_dir(img_dir)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_file() {
            continue;
        }

        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        if !["jpg", "jpeg", "png"].contains(&ext.as_str()) {
            continue;
        }

        let filename = path.file_stem().unwrap().to_string_lossy();
        println!(
            "Processing {}...",
            path.file_name().unwrap().to_string_lossy()
        );

        let img = image::open(&path)?;

        // Detect faces
        let faces = detector.detect(&img)?;
        println!("  -> Found {} faces", faces.len());

        for (i, face) in faces.iter().enumerate() {
            // Alignment requires the 5 facial landmarks (eyes, nose, mouth corners)
            if let Some(landmarks) = &face.landmarks {
                if landmarks.len() != 5 {
                    println!(
                        "  -> Face {} skipped: expected 5 landmarks, found {}",
                        i,
                        landmarks.len()
                    );
                    continue;
                }

                // Convert Vec to fixed array [ (f32, f32); 5 ] required by norm_crop
                let lms_array: [(f32, f32); 5] = [
                    landmarks[0],
                    landmarks[1],
                    landmarks[2],
                    landmarks[3],
                    landmarks[4],
                ];

                // 4. Perform Face Alignment
                // 112 is the standard size for ArcFace/InsightFace models
                let aligned_img = norm_crop(&img, &lms_array, 112);

                // 5. Save the result
                let out_name = format!("{}_face_{}.png", filename, i);
                let out_path = Path::new(output_base).join(out_name);
                aligned_img.save(&out_path)?;
                println!("  -> Saved aligned face to: {:?}", out_path);
            } else {
                println!(
                    "  -> Face {} skipped: No landmarks detected (is this a 'kps' model?)",
                    i
                );
            }
        }
    }

    println!("\nDone! Aligned faces are in '{}'", output_base);
    Ok(())
}
