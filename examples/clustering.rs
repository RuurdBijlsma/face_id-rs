#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use color_eyre::eyre::Result;
use face_id::analyzer::{FaceAnalysis, FaceAnalyzer};
use face_id::helpers::{cluster_faces, extract_face_thumbnail};
use image::RgbImage;
use ort::ep::CUDA;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use walkdir::WalkDir;

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let input_dir = "assets/img";
    let output_base = Path::new("output_previews/clusters");
    if output_base.exists() {
        fs::remove_dir_all(output_base)?;
    }
    fs::create_dir_all(output_base)?;

    println!("Initializing models...");
    let analyzer = FaceAnalyzer::from_hf()
        .with_execution_providers(&[CUDA::default().build().error_on_failure()])
        .build()
        .await?;

    println!("Scanning directory: {input_dir}");
    let image_paths: Vec<PathBuf> = WalkDir::new(input_dir)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .map(|e| e.path().to_path_buf())
        .filter(|p| is_image(p))
        .collect();

    if image_paths.is_empty() {
        println!("No images found.");
        return Ok(());
    }

    println!(
        "Found {} images. Starting analysis and clustering...",
        image_paths.len()
    );
    let now = Instant::now();
    // Use the high-level helper to analyze and cluster faces
    let clusters: HashMap<i32, Vec<(PathBuf, FaceAnalysis)>> =
        cluster_faces(&analyzer, image_paths)
            .min_cluster_size(5)
            .call()?;
    let face_count = clusters.iter().fold(0, |acc, (_, faces)| acc + faces.len());
    println!(
        "cluster_faces for {} faces took {:?}",
        face_count,
        now.elapsed()
    );

    if clusters.is_empty() {
        println!("No clusters found.");
        return Ok(());
    }

    println!("Writing cluster grid images to disk...");
    clusters
        .par_iter()
        .for_each(|(label, members): (&i32, &Vec<(PathBuf, FaceAnalysis)>)| {
            let cluster_name = if *label == -1 {
                "noise".to_string()
            } else {
                format!("cluster_{label}")
            };

            let thumb_size = 256;
            let padding = 1.6;
            let count = members.len();
            if count == 0 {
                return;
            }

            // Determine grid dimensions
            let cols = (count as f32).sqrt().ceil() as u32;
            let rows = (count as f32 / cols as f32).ceil() as u32;

            let mut grid_img = RgbImage::new(cols * thumb_size, rows * thumb_size);

            for (member_idx, (path, face)) in members.iter().enumerate() {
                let img = match image::open(path) {
                    Ok(i) => i,
                    Err(e) => {
                        eprintln!("Failed to open {}: {}", path.display(), e);
                        continue;
                    }
                };

                // Extract a high-quality thumbnail for the face
                let thumbnail: RgbImage = extract_face_thumbnail(
                    &img,
                    &face.detection.bbox,
                    padding,
                    thumb_size,
                );

                let x = (member_idx as u32 % cols) * thumb_size;
                let y = (member_idx as u32 / cols) * thumb_size;

                image::imageops::replace(&mut grid_img, &thumbnail, x as i64, y as i64);
            }

            let out_name = format!("{cluster_name}.jpg");
            grid_img.save(output_base.join(out_name)).unwrap();
        });

    println!("Done! Check output_previews/clusters");
    Ok(())
}

fn is_image(path: &Path) -> bool {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png")
}
