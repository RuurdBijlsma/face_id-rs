use color_eyre::eyre::Result;
use face_id::detector::ScrfdDetector;
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_rect_mut};
use imageproc::rect::Rect;
use std::fs;
use std::path::Path;

fn main() -> Result<()> {
    color_eyre::install()?;

    let model_path = "assets/models/scrfd_10g_bnkps.onnx";
    let img_dir = "assets/img";
    let output_dir = "output_previews";

    // Create output directory if it doesn't exist
    if !Path::new(output_dir).exists() {
        fs::create_dir_all(output_dir)?;
    }

    // Initialize detector
    let mut detector = ScrfdDetector::builder(model_path).build()?;

    // Iterate over files in the image directory
    for entry in fs::read_dir(img_dir)? {
        let entry = entry?;
        let path = entry.path();

        // Check file extension
        let extension = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        if !["jpg", "jpeg", "png"].contains(&extension.as_str()) {
            continue;
        }

        let filename = path.file_name().unwrap().to_string_lossy();
        println!("Processing {}...", filename);

        // 1. Load Image
        let img = image::open(&path)?;

        // 2. Detect Faces
        let faces = detector.detect(&img)?;
        println!("  -> Found {} faces", faces.len());

        // 3. Prepare for drawing (convert to RGB8)
        let mut output_img: RgbImage = img.to_rgb8();

        for face in faces {
            let b = face.bbox;

            // Draw Bounding Box (Green)
            // Ensure values are within reasonable bounds for i32/u32
            let x = b.x1.max(0.0) as i32;
            let y = b.y1.max(0.0) as i32;
            let w = b.width().max(0.0) as u32;
            let h = b.height().max(0.0) as u32;

            draw_hollow_rect_mut(
                &mut output_img,
                Rect::at(x, y).of_size(w, h),
                Rgb([0, 255, 0]), // Green
            );

            // Draw Landmarks (Red)
            if let Some(landmarks) = face.landmarks {
                for pt in landmarks {
                    draw_filled_circle_mut(
                        &mut output_img,
                        (pt.0 as i32, pt.1 as i32),
                        2,
                        Rgb([255, 0, 0]), // Red
                    );
                }
            }
        }

        // 4. Save Output
        let output_path = Path::new(output_dir).join(format!("detected_{}", filename));
        output_img.save(&output_path)?;
    }

    println!("\nDone! Check the '{}' folder for results.", output_dir);
    Ok(())
}
