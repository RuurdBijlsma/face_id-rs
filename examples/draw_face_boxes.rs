use ab_glyph::{FontRef, PxScale};
use color_eyre::eyre::Result;
use face_id::analyzer::FaceAnalyzer;
use face_id::gender_age::Gender;
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use std::fs;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    // 1. Setup paths
    let img_dir = "assets/img";
    let output_img_dir = Path::new("output_previews");
    let output_json_path = "assets/reference_analysis.json";

    if !output_img_dir.exists() {
        fs::create_dir_all(output_img_dir)?;
    }

    // 2. Initialize the Comprehensive Analyzer from Hugging Face
    println!("🚀 Loading models from Hugging Face (Detector, Embedder, Gender/Age)...");
    let analyzer = FaceAnalyzer::from_hf().build().await?;

    // 3. Load font for drawing text labels
    let font_data = include_bytes!("../assets/font/Roboto-Medium.ttf");
    let font = FontRef::try_from_slice(font_data)?;

    let mut all_results = Vec::new();

    // 4. Process all images in the directory
    println!("📂 Processing images in: {}", img_dir);
    for entry in fs::read_dir(img_dir)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_file() {
            continue;
        }

        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if !["jpg", "jpeg", "png"].contains(&ext.to_lowercase().as_str()) {
            continue;
        }

        let filename = path.file_name().unwrap().to_string_lossy().to_string();
        println!("\n📸 Analyzing: {}", filename);

        let img = image::open(&path)?;
        let mut output_img: RgbImage = img.to_rgb8();

        // RUN PIPELINE: Detection -> Alignment -> Embedder -> Attributes
        let analysis_results = analyzer.analyze(&img)?;
        println!("  ✅ Found {} face(s)", analysis_results.len());

        for (i, face) in analysis_results.iter().enumerate() {
            let det = &face.detection;

            // Print Info to Stdout
            let ga_str = if let Some(ga) = &face.gender_age {
                format!("{:?} (Age: {})", ga.gender, ga.age)
            } else {
                "N/A".to_string()
            };

            let emb_preview = if let Some(emb) = &face.embedding {
                format!("{:?}", &emb[..5]) // First 5 dims
            } else {
                "N/A".to_string()
            };

            println!(
                "      [Face #{}] Score: {:.2} | GA: {} | Embedding (5 dims): {}...",
                i, det.score, ga_str, emb_preview
            );

            // --- DRAWING ---
            let b = det.bbox;
            let x = b.x1.max(0.0) as i32;
            let y = b.y1.max(0.0) as i32;
            let w = b.width().max(0.0) as u32;
            let h = b.height().max(0.0) as u32;

            // Pick color based on gender if available
            let color = match face.gender_age.as_ref().map(|ga| ga.gender) {
                Some(Gender::Male) => Rgb([0, 150, 255]),   // Blue
                Some(Gender::Female) => Rgb([255, 105, 180]), // Pink
                None => Rgb([0, 255, 0]),                    // Green
            };

            // Draw Bounding Box
            draw_hollow_rect_mut(&mut output_img, Rect::at(x, y).of_size(w, h), color);

            // Draw Landmarks (Red Dots)
            if let Some(landmarks) = &det.landmarks {
                for pt in landmarks {
                    draw_filled_circle_mut(
                        &mut output_img,
                        (pt.0 as i32, pt.1 as i32),
                        2,
                        Rgb([255, 0, 0]),
                    );
                }
            }

            // Draw Label (Gender/Age)
            if let Some(ga) = &face.gender_age {
                let gender_label = match ga.gender {
                    Gender::Male => "M",
                    Gender::Female => "F",
                };
                let label = format!("{}, {}", gender_label, ga.age);
                let scale = PxScale::from(20.0);
                draw_text_mut(
                    &mut output_img,
                    color,
                    x,
                    (y - 25).max(0),
                    scale,
                    &font,
                    &label,
                );
            }
        }

        // Save annotated image
        let out_path = output_img_dir.join(format!("analysis_{}", filename));
        output_img.save(&out_path)?;

        // Collect data for JSON
        all_results.push(serde_json::json!({
            "filename": filename,
            "results": analysis_results
        }));
    }

    // 5. Generate Reference JSON
    let file = fs::File::create(output_json_path)?;
    serde_json::to_writer_pretty(file, &all_results)?;

    println!("\n✨ Process complete!");
    println!("🖼️  Annotated images: {:?}", output_img_dir);
    println!("📄 Reference JSON: {}", output_json_path);

    Ok(())
}