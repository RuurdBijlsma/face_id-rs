#![allow(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

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

    let img_dir = "assets/img";
    let output_img_dir = Path::new("output_previews");
    let output_json_path = "assets/reference_analysis.json";

    if !output_img_dir.exists() {
        fs::create_dir_all(output_img_dir)?;
    }
    println!("Loading models (Detector, Embedder, Gender/Age)...");
    let analyzer = FaceAnalyzer::from_hf().build().await?;
    let font_data = include_bytes!("../assets/font/Roboto-Medium.ttf");
    let font = FontRef::try_from_slice(font_data)?;

    let mut all_results = Vec::new();
    println!("Processing images in: {img_dir}");
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
        println!("\nAnalyzing: {filename}");

        let img = image::open(&path)?;
        let mut output_img: RgbImage = img.to_rgb8();
        let analysis_results = analyzer.analyze(&img)?;
        println!("  Found {} face(s)", analysis_results.len());

        for (i, face) in analysis_results.iter().enumerate() {
            let det = face.detection.to_absolute(img.width(), img.height());
            let ga_str = face.gender_age.as_ref().map_or_else(
                || "N/A".to_string(),
                |ga| format!("{:?} (Age: {})", ga.gender, ga.age),
            );

            let emb_preview = face
                .embedding
                .as_ref()
                .map_or_else(|| "N/A".to_string(), |emb| format!("{:?}", &emb[..5]));

            println!(
                "      [Face #{}] Score: {:.2} | GA: {} | Embedding (5 dims): {}...",
                i, det.score, ga_str, emb_preview
            );

            // --- DRAWING ---
            let b = det.bbox;
            let x = b.x1 as i32;
            let y = b.y1 as i32;
            let w = b.width() as u32;
            let h = b.height() as u32;
            let color = match face.gender_age.as_ref().map(|ga| ga.gender) {
                Some(Gender::Male) => Rgb([0, 150, 255]),     // Blue
                Some(Gender::Female) => Rgb([255, 105, 180]), // Pink
                None => Rgb([0, 255, 0]),                     // Green
            };
            draw_hollow_rect_mut(&mut output_img, Rect::at(x, y).of_size(w, h), color);
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

        let out_path = output_img_dir.join(format!("analysis_{filename}"));
        output_img.save(&out_path)?;
        all_results.push(serde_json::json!({
            "filename": filename,
            "results": analysis_results
        }));
    }

    let file = fs::File::create(output_json_path)?;
    serde_json::to_writer_pretty(file, &all_results)?;

    println!("\nProcess complete!");
    println!("Annotated images: {}", output_img_dir.display());
    println!("Reference JSON: {output_json_path}");

    Ok(())
}
