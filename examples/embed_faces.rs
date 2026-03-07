use color_eyre::eyre::Result;
use face_id::detector::ScrfdDetector;
use face_id::face_align::norm_crop;
use face_id::recognizer::ArcFaceEmbedder;
use std::fs;

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    // 1. Configuration
    let img_dir = "assets/img";

    // 2. Initialize Detector and Recognizer
    println!("Loading models from Hugging Face...");
    let mut detector = ScrfdDetector::from_hf().build().await?;
    let mut recognizer = ArcFaceEmbedder::from_hf().build().await?;

    println!("Models loaded successfully.");

    // 3. Process images
    for entry in fs::read_dir(img_dir)? {
        let entry = entry?;
        let path = entry.path();

        println!("\nProcessing: {:?}", path.file_name().unwrap());

        let img = image::open(&path)?;

        // A. Detect Faces
        let faces = detector.detect(&img)?;
        println!("  Found {} face(s)", faces.len());

        for (i, face) in faces.iter().enumerate() {
            // Check if we have the 5 keypoints required for alignment
            if let Some(landmarks) = &face.landmarks {
                if landmarks.len() != 5 {
                    continue;
                }

                // B. Align Face
                // Convert landmarks to the fixed array format required by norm_crop
                let lms_array: [(f32, f32); 5] = [
                    landmarks[0],
                    landmarks[1],
                    landmarks[2],
                    landmarks[3],
                    landmarks[4],
                ];

                // 112x112 is the standard input size for ArcFace
                let aligned_face = norm_crop(&img, &lms_array, 112);

                // C. Compute Embedding
                let embedding = recognizer.compute_embedding(&aligned_face)?;

                // D. Output result
                println!(
                    "  Face #{} [Score: {:.2}]: Embedding (first 5 dims): {:?}",
                    i,
                    face.score,
                    &embedding[..5]
                );
                println!("  Embedding size: {} dimensions", embedding.len());
            } else {
                println!("  Face #{} skipped: No landmarks detected.", i);
            }
        }
    }

    Ok(())
}
