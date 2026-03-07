use color_eyre::eyre::Result;
use face_id::detector::ScrfdDetector;
use face_id::gender_age::{Gender, GenderAgeEstimator};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use std::fs;
use std::path::Path;
use ab_glyph::{FontRef, PxScale};

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    // 1. Configuration
    let det_model_id = "RuteNL/SCRFD-face-detection-ONNX";
    let det_model_file = "34g_gnkps.onnx";

    // Standard InsightFace Buffalo_L gender/age model
    let attr_model_id = "public-data/insightface";
    let attr_model_file = "models/buffalo_l/genderage.onnx";

    let img_dir = "assets/img";
    let output_dir = Path::new("output_previews/gender_age");

    // 2. Initialize Models
    println!("Loading models...");
    let mut detector = ScrfdDetector::from_hf(det_model_id, det_model_file)
        .build()
        .await?;

    let mut estimator = GenderAgeEstimator::from_hf(attr_model_id, attr_model_file)
        .build()
        .await?;

    if !output_dir.exists() {
        fs::create_dir_all(output_dir)?;
    }

    let font_data = include_bytes!("../assets/font/Roboto-Medium.ttf");
    let font = FontRef::try_from_slice(font_data)?;

    println!("Processing images in {}...", img_dir);

    // 3. Process Images
    for entry in fs::read_dir(img_dir)? {
        let entry = entry?;
        let path = entry.path();

        let filename = path.file_name().unwrap().to_string_lossy();
        println!("\nProcessing {}...", filename);

        let img = image::open(&path)?;
        let mut output_img: RgbImage = img.to_rgb8();

        // A. Detect Faces
        let faces = detector.detect(&img)?;
        println!("  Found {} face(s)", faces.len());

        for (i, face) in faces.iter().enumerate() {
            // B. Estimate Gender and Age
            let result = estimator.estimate(&img, &face.bbox)?;

            let gender_str = match result.gender {
                Gender::Male => "M",
                Gender::Female => "F",
            };

            println!(
                "  Face #{}: Gender={:?}, Age={}",
                i, result.gender, result.age
            );

            // C. Draw Results
            let b = face.bbox;
            let x = b.x1.max(0.0) as i32;
            let y = b.y1.max(0.0) as i32;
            let w = b.width().max(0.0) as u32;
            let h = b.height().max(0.0) as u32;

            // Color code: Blue for Male, Pink/Red for Female
            let color = match result.gender {
                Gender::Male => Rgb([0, 150, 255]),
                Gender::Female => Rgb([255, 105, 180]),
            };

            // Draw bounding box
            draw_hollow_rect_mut(
                &mut output_img,
                Rect::at(x, y).of_size(w, h),
                color,
            );

            // Draw Label: "M, 25"
            let label = format!("{}, {}", gender_str, result.age);
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

        // D. Save output
        let out_path = output_dir.join(format!("attr_{}", filename));
        output_img.save(&out_path)?;
    }

    println!("\nDone! Results saved to {:?}", output_dir);
    Ok(())
}