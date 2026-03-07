use color_eyre::eyre::Result;
use face_id::detector::ScrfdDetector;
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_rect_mut};
use imageproc::rect::Rect;
use std::fs;
use std::path::Path;

fn main() -> Result<()> {
    color_eyre::install()?;

    let models_dir = "assets/models";
    let img_dir = "assets/img";
    let base_output_dir = "output_previews/draw_faces";

    // Iterate over all models
    for model_entry in fs::read_dir(models_dir)? {
        let model_entry = model_entry?;
        let model_path = model_entry.path();
        
        let model_name = model_path.file_stem().unwrap().to_string_lossy();
        println!("\n==================================");
        println!("Testing model: {}", model_name);
        println!("==================================");

        let output_dir = Path::new(base_output_dir).join(&*model_name);
        if !output_dir.exists() {
            fs::create_dir_all(&output_dir)?;
        }

        // Initialize detector
        let mut detector = ScrfdDetector::builder(&model_path).build()?;

        // Iterate over files in the image directory
        for img_entry in fs::read_dir(img_dir)? {
            let img_entry = img_entry?;
            let path = img_entry.path();

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

            let img = image::open(&path)?;

            let faces = detector.detect(&img) ?;
            println!("  -> Found {} faces", faces.len());

            let mut output_img: RgbImage = img.to_rgb8();

            for face in faces {
                let b = face.bbox;
                let x = b.x1.max(0.0) as i32;
                let y = b.y1.max(0.0) as i32;
                let w = b.width().max(0.0) as u32;
                let h = b.height().max(0.0) as u32;

                draw_hollow_rect_mut(
                    &mut output_img,
                    Rect::at(x, y).of_size(w, h),
                    Rgb([0, 255, 0]),
                );

                if let Some(landmarks) = face.landmarks {
                    for pt in landmarks {
                        draw_filled_circle_mut(
                            &mut output_img,
                            (pt.0 as i32, pt.1 as i32),
                            2,
                            Rgb([255, 0, 0]),
                        );
                    }
                }
            }

            let output_path = output_dir.join(format!("detected_{}", filename));
            if let Err(e) = output_img.save(&output_path) {
                println!("  -> Failed to save output image: {}", e);
            }
        }
    }

    println!("\nDone! Check the '{}' folder for results.", base_output_dir);
    Ok(())
}
