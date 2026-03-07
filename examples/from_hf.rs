use color_eyre::eyre::Result;
use face_id::detector::ScrfdDetector;
use face_id::model_manager::HfModel;
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_rect_mut};
use imageproc::rect::Rect;
use std::fs;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let model_id = "RuteNL/SCRFD-face-detection-ONNX";
    let model_filenames = &[
        // "1g.onnx",
        // "2.5g_bnkps.onnx",
        // "10g_bnkps.onnx",
        // "34g.onnx",
        "34g_gnkps.onnx",
        // "500m.onnx",
    ];

    let img_dir = "assets/img";
    let base_output_dir = "output_previews";

    for model_filename in model_filenames {
        println!("\n==================================");
        println!("Testing model: {}", model_filename);
        println!("==================================");

        let output_dir = Path::new(base_output_dir).join(&*model_filename);
        if !output_dir.exists() {
            fs::create_dir_all(&output_dir)?;
        }

        // Initialize detector
        let mut detector = ScrfdDetector::from_hf()
            .model(HfModel {
                id: model_id.to_owned(),
                file: (*model_filename).to_owned(),
            })
            .build()
            .await?;

        // Iterate over files in the image directory
        for img_entry in fs::read_dir(img_dir)? {
            let img_entry = img_entry?;
            let path = img_entry.path();

            let filename = path.file_name().unwrap().to_string_lossy();
            println!("Processing {}...", filename);

            let img = image::open(&path)?;

            let faces = detector.detect(&img)?;
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

    println!(
        "\nDone! Check the '{}' folder for results.",
        base_output_dir
    );
    Ok(())
}
