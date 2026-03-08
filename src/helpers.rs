#![allow(clippy::similar_names)]
#[cfg(feature = "clustering")]
use crate::analyzer::{FaceAnalysis, FaceAnalyzer};
use crate::detector::BoundingBox;
#[cfg(feature = "clustering")]
use crate::error::FaceIdError;
#[cfg(feature = "clustering")]
use hdbscan::{DistanceMetric, Hdbscan, HdbscanHyperParams, NnAlgorithm};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
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

    // Determine the size of our square crop "side"
    let side = width.max(height) * padding_factor;

    // Calculate the theoretical coordinates of the crop
    let x1 = (cx - side / 2.0).round() as i32;
    let y1 = (cy - side / 2.0).round() as i32;
    let side_u = side.round() as u32;

    // Calculate intersection between the crop and the actual image
    let src_x1 = x1.max(0) as u32;
    let src_y1 = y1.max(0) as u32;
    let src_x2 = (x1 + side_u.cast_signed()).min(img_w.cast_signed()) as u32;
    let src_y2 = (y1 + side_u.cast_signed()).min(img_h.cast_signed()) as u32;

    if src_x2 > src_x1 && src_y2 > src_y1 {
        let crop_w = src_x2 - src_x1;
        let crop_h = src_y2 - src_y1;

        // Extract the valid part of the image
        let sub_img = img.view(src_x1, src_y1, crop_w, crop_h).to_image();

        // Use resize instead of resize_exact to maintain the aspect ratio if it's not square
        DynamicImage::ImageRgba8(sub_img)
            .resize(size, size, image::imageops::FilterType::CatmullRom)
            .to_rgb8()
    } else {
        // Fallback: This case shouldn't be hit with a valid bounding box that is within the image.
        // We'll return a small empty image to satisfy the return type.
        ImageBuffer::new(size, size)
    }
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
    let all_faces: Vec<(PathBuf, FaceAnalysis)> = paths
        .into_par_iter()
        .map(
            |path_ref| -> Result<Vec<(PathBuf, FaceAnalysis)>, FaceIdError> {
                let path = path_ref.as_ref().to_path_buf();
                let img = image::open(&path)?;
                let faces = analyzer.analyze(&img)?;
                Ok(faces.into_iter().map(|f| (path.clone(), f)).collect())
            },
        )
        .collect::<Result<Vec<Vec<_>>, _>>()?
        .into_iter()
        .flatten()
        .collect();

    if all_faces.is_empty() {
        return Ok(HashMap::new());
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_extract_face_thumbnail_edge_case() {
        // 50x100 white image
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(50, 100, Rgb([255, 255, 255])));

        // Face near the right edge, so with padding/aspect ratio 1 it would stick out
        let bbox = BoundingBox {
            x1: 40.0,
            y1: 50.0,
            x2: 50.0,
            y2: 60.0,
        };

        let thumbnail = extract_face_thumbnail(&img, &bbox, 4.0, 100);

        // Expected dimensions: fitting 25x40 into 100x100 results in ~63x100
        assert_ne!(thumbnail.width(), thumbnail.height());
        assert_eq!(thumbnail.width(), 63);
        assert_eq!(thumbnail.height(), 100);
    }
}
