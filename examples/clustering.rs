#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use color_eyre::eyre::Result;
use face_id::analyzer::FaceAnalyzer;
use face_id::detector::DetectedFace;
use hdbscan::Hdbscan;
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use walkdir::WalkDir;

struct FaceMetadata {
    path: PathBuf,
    detection: DetectedFace,
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let input_dir = "C:/Users/Ruurd/Pictures/media_dir";
    let output_base = Path::new("output_previews/clusters");
    if output_base.exists() {
        fs::remove_dir_all(output_base)?;
    }
    fs::create_dir_all(output_base)?;
    println!("Initializing models...");
    let analyzer = Arc::new(FaceAnalyzer::from_hf().build().await?);
    println!("Scanning directory: {input_dir}");
    let image_paths: Vec<PathBuf> = WalkDir::new(input_dir)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .map(|e| e.path().to_path_buf())
        .filter(|p| is_image(p))
        .collect();
    println!(
        "Found {} images. Starting parallel analysis...",
        image_paths.len()
    );

    let face_data: Vec<(Vec<f32>, FaceMetadata)> = image_paths
        .par_iter()
        .flat_map(|path| {
            let img = match image::open(path) {
                Ok(i) => i,
                Err(e) => {
                    eprintln!("Failed to open {}: {}", path.display(), e);
                    return Vec::new();
                }
            };
            let analysis_results = match analyzer.analyze(&img) {
                Ok(res) => res,
                Err(e) => {
                    eprintln!("Failed to analyze {}: {}", path.display(), e);
                    return Vec::new();
                }
            };
            analysis_results
                .into_iter()
                .filter_map(|face| {
                    let emb = face.embedding?;
                    Some((
                        emb,
                        FaceMetadata {
                            path: path.clone(),
                            detection: face.detection,
                        },
                    ))
                })
                .collect::<Vec<_>>()
        })
        .collect();

    if face_data.is_empty() {
        println!("No faces with embeddings found.");
        return Ok(());
    }
    let (embeddings, face_store): (Vec<Vec<f32>>, Vec<FaceMetadata>) =
        face_data.into_iter().unzip();

    println!("Clustering {} faces...", embeddings.len());
    let clusterer = Hdbscan::default_hyper_params(&embeddings);
    let labels = clusterer
        .cluster()
        .map_err(|e| color_eyre::eyre::eyre!(e))?;
    let mut clusters: HashMap<i32, Vec<(usize, &FaceMetadata)>> = HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        clusters
            .entry(label)
            .or_default()
            .push((idx, &face_store[idx]));
    }

    println!("Writing cluster results to disk...");
    clusters.par_iter().for_each(|(&label, members)| {
        let cluster_name = if label == -1 {
            "noise".to_string()
        } else {
            format!("cluster_{label}")
        };

        let cluster_dir = output_base.join(&cluster_name);
        fs::create_dir_all(&cluster_dir).unwrap();

        for (member_idx, metadata) in members {
            let img = image::open(&metadata.path).unwrap();
            let mut output_img: RgbImage = img.to_rgb8();

            let b = &metadata.detection.bbox;
            let rect =
                Rect::at(b.x1 as i32, b.y1 as i32).of_size(b.width() as u32, b.height() as u32);

            draw_hollow_rect_mut(&mut output_img, rect, Rgb([0, 255, 0]));

            let file_stem = metadata.path.file_stem().unwrap().to_string_lossy();
            let out_name = format!("{file_stem}_face_{member_idx}.jpg");
            output_img.save(cluster_dir.join(out_name)).unwrap();
        }
    });

    println!("Done! Check output_previews/clusters");
    Ok(())
}

fn is_image(path: &Path) -> bool {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png")
}
