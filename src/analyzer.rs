use crate::detector::{DetectedFace, ScrfdDetector};
use crate::embedder::ArcFaceEmbedder;
use crate::error::FaceIdError;
use crate::face_align::norm_crop;
use crate::gender_age::{GenderAge, GenderAgeEstimator};
use crate::model_manager::HfModel;
use bon::bon;
use image::DynamicImage;
#[cfg(feature = "cuda")]
use ort::ep::CUDA;
use ort::ep::ExecutionProviderDispatch;
use rayon::prelude::*;
use std::path::Path;
use std::sync::Mutex;

#[cfg(feature = "cuda")]
pub fn default_optimized_cuda() -> ExecutionProviderDispatch {
    CUDA::default()
        .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Exhaustive)
        .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::NextPowerOfTwo)
        .with_conv_max_workspace(true)
        .with_conv1d_pad_to_nc1d(false)
        .with_tf32(true)
        .build()
        .error_on_failure()
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FaceAnalysis {
    pub detection: DetectedFace,
    pub embedding: Option<Vec<f32>>,
    pub gender_age: Option<GenderAge>,
}

/// Performs detection, alignment, embedding, and gender/age estimation.
pub struct FaceAnalyzer {
    pub detector: Mutex<ScrfdDetector>,
    pub embedder: Mutex<ArcFaceEmbedder>,
    pub gender_age: Mutex<GenderAgeEstimator>,
}

#[bon]
impl FaceAnalyzer {
    #[builder(finish_fn = build)]
    pub async fn from_hf(
        #[builder(default = HfModel::default_detector())] detector_model: HfModel,
        #[builder(default = HfModel::default_embedder())] embedder_model: HfModel,
        #[builder(default = HfModel::default_gender_age())] gender_age_model: HfModel,
        #[builder(default = (640, 640))] detector_input_size: (u32, u32),
        #[builder(default = 0.5)] detector_score_threshold: f32,
        #[builder(default = 0.4)] detector_iou_threshold: f32,
        #[builder(default = &[])] with_execution_providers: &[ExecutionProviderDispatch],
    ) -> Result<Self, FaceIdError> {
        let detector = ScrfdDetector::from_hf()
            .input_size(detector_input_size)
            .score_threshold(detector_score_threshold)
            .iou_threshold(detector_iou_threshold)
            .model(detector_model)
            .with_execution_providers(with_execution_providers)
            .build()
            .await?;
        let embedder = ArcFaceEmbedder::from_hf()
            .model(embedder_model)
            .with_execution_providers(with_execution_providers)
            .build()
            .await?;
        let gender_age = GenderAgeEstimator::from_hf()
            .model(gender_age_model)
            .with_execution_providers(with_execution_providers)
            .build()
            .await?;

        Ok(Self {
            detector: Mutex::new(detector),
            embedder: Mutex::new(embedder),
            gender_age: Mutex::new(gender_age),
        })
    }

    /// Creates a new analyzer using local paths to ONNX model files.
    #[builder(finish_fn = build)]
    pub fn new(
        #[builder(start_fn)] det_model: impl AsRef<Path>,
        #[builder(start_fn)] rec_model: impl AsRef<Path>,
        #[builder(start_fn)] attr_model: impl AsRef<Path>,
        #[builder(default = (640, 640))] detector_input_size: (u32, u32),
        #[builder(default = 0.5)] detector_score_threshold: f32,
        #[builder(default = 0.4)] detector_iou_threshold: f32,
        #[builder(default = &[])] with_execution_providers: &[ExecutionProviderDispatch],
    ) -> Result<Self, FaceIdError> {
        let detector = ScrfdDetector::builder(det_model)
            .input_size(detector_input_size)
            .score_threshold(detector_score_threshold)
            .iou_threshold(detector_iou_threshold)
            .with_execution_providers(with_execution_providers)
            .build()?;

        let embedder = ArcFaceEmbedder::builder(rec_model)
            .with_execution_providers(with_execution_providers)
            .build()?;

        let gender_age = GenderAgeEstimator::builder(attr_model)
            .with_execution_providers(with_execution_providers)
            .build()?;

        Ok(Self {
            detector: Mutex::new(detector),
            embedder: Mutex::new(embedder),
            gender_age: Mutex::new(gender_age),
        })
    }

    /// Performs the full pipeline: detection -> alignment -> embedding -> gender/age estimation.
    pub fn analyze(&self, img: &DynamicImage) -> Result<Vec<FaceAnalysis>, FaceIdError> {
        let rgb_img = img.to_rgb8();

        // Detect face bounding boxes & landmarks
        let mut results = self
            .detector
            .lock()
            .map_err(|_| FaceIdError::MutexPoisoned("Detector".into()))?
            .detect(img)?
            .into_iter()
            .map(|det| FaceAnalysis {
                detection: det,
                embedding: None,
                gender_age: None,
            })
            .collect::<Vec<_>>();

        if results.is_empty() {
            return Ok(vec![]);
        }

        // Embedding: Alignment & Batch Inference
        let (embed_crops, embed_indices): (Vec<_>, Vec<_>) = results
            .par_iter()
            .enumerate()
            .filter_map(|(idx, res)| {
                let landmarks = res.detection.landmarks.as_ref()?;
                let lms_array: [(f32, f32); 5] = landmarks.as_slice().try_into().ok()?;
                let aligned = norm_crop(&rgb_img, &lms_array, 112);
                Some((aligned, idx))
            })
            .unzip();

        if !embed_crops.is_empty() {
            let embeddings = self
                .embedder
                .lock()
                .map_err(|_| FaceIdError::MutexPoisoned("Embedder".into()))?
                .compute_embeddings_batch(&embed_crops)?;
            for (emb, original_idx) in embeddings.into_iter().zip(embed_indices) {
                results[original_idx].embedding = Some(emb);
            }
        }

        // Gender/Age: Alignment & Batch Inference
        let (ga_crops, ga_indices): (Vec<_>, Vec<_>) = results
            .par_iter()
            .enumerate()
            .map(|(idx, res)| {
                let crop = GenderAgeEstimator::align_crop(&rgb_img, &res.detection.bbox, 96);
                (crop, idx)
            })
            .unzip();

        if !ga_crops.is_empty() {
            let ga_results = self
                .gender_age
                .lock()
                .map_err(|_| FaceIdError::MutexPoisoned("GenderAge".into()))?
                .estimate_batch(&ga_crops)?;
            for (ga_val, original_idx) in ga_results.into_iter().zip(ga_indices) {
                results[original_idx].gender_age = Some(ga_val);
            }
        }

        Ok(results)
    }
}
