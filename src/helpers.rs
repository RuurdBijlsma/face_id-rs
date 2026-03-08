#![allow(clippy::similar_names)]
use crate::detector::BoundingBox;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
#[cfg(feature = "clustering")]
use crate::analyzer::{FaceAnalysis, FaceAnalyzer};
#[cfg(feature = "clustering")]
use crate::error::FaceIdError;
#[cfg(feature = "clustering")]
use hdbscan::{DistanceMetric, Hdbscan, HdbscanHyperParams, NnAlgorithm};
use rayon::prelude::*;
#[cfg(feature = "clustering")]
use std::collections::HashMap;
#[cfg(feature = "clustering")]
use std::path::{Path, PathBuf};

/// Extracts a square, padded thumbnail for a face.
///
/// # Arguments
/// * `img` - The source image.
/// * `bbox` - The detected face bounding box.
/// * `padding_factor` - How much context to show. 1.0 = tight crop, 2.0 = face takes up 50% of width. (1.5 - 1.8 is usually ideal for UI).
/// * `size` - The output resolution (e.g., 256 for a 256x256 thumbnail).
#[must_use]
pub fn extract_face_thumbnail(
    img: &DynamicImage,
    bbox: &BoundingBox,
    padding_factor: f32,
    size: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (img_w, img_h) = img.dimensions();

    let width = bbox.width();
    let height = bbox.height();
    let cx = bbox.x1 + width / 2.0;
    let cy = bbox.y1 + height / 2.0;

    // 1. Determine the size of our square crop "side"
    // We use the largest dimension to ensure the face fits if it's tall or wide
    let side = width.max(height) * padding_factor;

    // 2. Calculate the theoretical coordinates of the crop
    let x1 = (cx - side / 2.0).round() as i32;
    let y1 = (cy - side / 2.0).round() as i32;
    let side_u = side.round() as u32;

    // 3. Create a black (or transparent) canvas of the requested 'side' size
    // Using Rgb here to match your crate's patterns, but Rgba could be used for transparency
    let mut canvas = ImageBuffer::new(side_u, side_u);

    // 4. Calculate intersection between the crop and the actual image
    // This handles the "cut off" requirement.
    let src_x1 = x1.max(0) as u32;
    let src_y1 = y1.max(0) as u32;
    let src_x2 = (x1 + side_u.cast_signed()).min(img_w.cast_signed()) as u32;
    let src_y2 = (y1 + side_u.cast_signed()).min(img_h.cast_signed()) as u32;

    if src_x2 > src_x1 && src_y2 > src_y1 {
        let crop_w = src_x2 - src_x1;
        let crop_h = src_y2 - src_y1;

        // Extract the valid part of the image
        let sub_img = img.view(src_x1, src_y1, crop_w, crop_h);

        // 5. Calculate where to paste the image onto the canvas
        // If x1 was negative, the offset will be positive
        let dst_x = (src_x1.cast_signed() - x1) as u64;
        let dst_y = (src_y1.cast_signed() - y1) as u64;

        // Overlay the valid pixels onto the black canvas
        image::imageops::overlay(
            &mut canvas,
            &sub_img.to_image(),
            dst_x.cast_signed(),
            dst_y.cast_signed(),
        );
    }

    // 6. Final resize to the standard thumbnail size
    let dynamic_canvas = DynamicImage::ImageRgba8(canvas);
    dynamic_canvas
        .resize_exact(size, size, image::imageops::FilterType::CatmullRom)
        .to_rgb8()
}

/// Clusters faces from a list of images using the HDBSCAN algorithm.
///
/// This function performs the following steps:
/// 1. Loads each image from the provided paths.
/// 2. Performs face analysis (detection and embedding) using the provided `FaceAnalyzer`.
/// 3. Clusters the resulting face embeddings using HDBSCAN.
///
/// Returns a mapping of cluster IDs to a list of (image path, face analysis) pairs.
/// Cluster ID -1 represents noise.
///
/// # Errors
/// Returns a `FaceIdError` if any image fails to load, analysis fails, or clustering fails.
///
/// # Feature Gated
/// This function is only available when the `clustering` feature is enabled.
#[cfg(feature = "clustering")]
#[bon::builder]
pub fn cluster_faces<P: AsRef<Path> + Sync + Send>(
    #[builder(start_fn)] analyzer: &FaceAnalyzer,
    #[builder(start_fn)] paths: Vec<P>,
    #[builder(default = 5)] min_cluster_size: usize,
    #[builder(default = usize::MAX)] max_cluster_size: usize,
    #[builder(default = false)] allow_single_cluster: bool,
    min_samples: Option<usize>,
    #[builder(default = 0.0)] epsilon: f64,
    #[builder(default = DistanceMetric::Euclidean)] dist_metric: DistanceMetric,
    #[builder(default = NnAlgorithm::Auto)] nn_algo: NnAlgorithm,
) -> Result<HashMap<i32, Vec<(PathBuf, FaceAnalysis)>>, FaceIdError> {
    // 1. Analyze images in parallel
    let all_faces: Vec<(PathBuf, FaceAnalysis)> = paths
        .into_par_iter()
        .map(|path_ref| -> Result<Vec<(PathBuf, FaceAnalysis)>, FaceIdError> {
            let path = path_ref.as_ref().to_path_buf();
            let img = image::open(&path)?;
            let faces = analyzer.analyze(&img)?;
            Ok(faces.into_iter().map(|f| (path.clone(), f)).collect())
        })
        .collect::<Result<Vec<Vec<_>>, _>>()?
        .into_iter()
        .flatten()
        .collect();

    if all_faces.is_empty() {
        return Ok(HashMap::new());
    }

    // 2. Prepare embeddings for clustering
    let (embeddings, face_refs): (Vec<Vec<f32>>, Vec<&(PathBuf, FaceAnalysis)>) = all_faces
        .iter()
        .filter_map(|pair| {
            pair.1
                .embedding
                .as_ref()
                .map(|emb: &Vec<f32>| (emb.clone(), pair))
        })
        .unzip();

    if embeddings.is_empty() {
        return Ok(HashMap::new());
    }

    // 3. Perform clustering
    let mut hp_builder = HdbscanHyperParams::builder()
        .min_cluster_size(min_cluster_size)
        .max_cluster_size(max_cluster_size)
        .allow_single_cluster(allow_single_cluster)
        .epsilon(epsilon)
        .dist_metric(dist_metric)
        .nn_algorithm(nn_algo);

    if let Some(ms) = min_samples {
        hp_builder = hp_builder.min_samples(ms);
    } else {
        hp_builder = hp_builder.min_samples(min_cluster_size);
    }

    let hyper_params = hp_builder.build();
    let clusterer = Hdbscan::new(&embeddings, hyper_params);
    let labels: Vec<i32> = clusterer
        .cluster()
        .map_err(|e| FaceIdError::Clustering(e.to_string()))?;

    // 4. Group results
    let mut clusters: HashMap<i32, Vec<(PathBuf, FaceAnalysis)>> = HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        let (path, face) = face_refs[idx];
        clusters
            .entry(label)
            .or_default()
            .push((path.clone(), face.clone()));
    }

    Ok(clusters)
}
