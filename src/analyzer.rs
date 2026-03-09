use crate::detector::{DetectedFace, ScrfdDetector};
use crate::embedder::ArcFaceEmbedder;
use crate::error::FaceIdError;
use crate::face_align::norm_crop;
use crate::gender_age::{Gender, GenderAgeEstimator};
use crate::model_manager::HfModel;
use bon::bon;
use image::DynamicImage;
use ort::ep::ExecutionProviderDispatch;
use rayon::prelude::*;
use std::path::Path;
use std::sync::Mutex;

/// Result of face analysis.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FaceAnalysis {
    /// The original detection details (bbox, landmarks, score).
    pub detection: DetectedFace,
    /// A 512-dimensional normalized embedding vector.
    pub embedding: Vec<f32>,
    /// Estimated gender.
    pub gender: Gender,
    /// Estimated age in years.
    pub age: u8,
}

/// Performs detection, alignment, embedding, and gender/age estimation.
///
/// This struct wraps multiple ONNX Runtime sessions. These sessions are wrapped in `Mutex`
/// because their inference methods requires mutable access. Accessing the analyzer from
/// multiple threads will involve locking.
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
        let results = self
            .detector
            .lock()
            .map_err(|_| FaceIdError::MutexPoisoned("Detector".into()))?
            .detect(img)?;

        if results.is_empty() {
            return Ok(vec![]);
        }

        let embed_crops: Vec<_> = results
            .par_iter()
            .map(|res| {
                let landmarks = res.landmarks.as_ref().ok_or_else(|| {
                    FaceIdError::InvalidModel(
                        "One or more faces missing landmarks for embedding".into(),
                    )
                })?;
                let lms_array: [(f32, f32); 5] = landmarks
                    .iter()
                    .map(|&(x, y)| (x * rgb_img.width() as f32, y * rgb_img.height() as f32))
                    .collect::<Vec<_>>()
                    .try_into()
                    .map_err(|_| {
                        FaceIdError::InvalidModel("Landmarks were not 5-point keypoints".into())
                    })?;
                Ok(norm_crop(&rgb_img, &lms_array, 112))
            })
            .collect::<Result<Vec<_>, FaceIdError>>()?;

        let embeddings = self
            .embedder
            .lock()
            .map_err(|_| FaceIdError::MutexPoisoned("Embedder".into()))?
            .compute_embeddings_batch(&embed_crops)?;

        // Gender/Age: Alignment & Batch Inference
        let ga_crops: Vec<_> = results
            .par_iter()
            .map(|res| GenderAgeEstimator::align_crop(&rgb_img, &res.bbox, 96))
            .collect();

        let ga_results = self
            .gender_age
            .lock()
            .map_err(|_| FaceIdError::MutexPoisoned("GenderAge".into()))?
            .estimate_batch(&ga_crops)?;

        if embeddings.len() != results.len() || ga_results.len() != results.len() {
            return Err(FaceIdError::Ort(format!(
                "Inconsistent batch results: expected {}, got {} embeddings and {} ga results",
                results.len(),
                embeddings.len(),
                ga_results.len()
            )));
        }

        let final_results = results
            .into_iter()
            .zip(embeddings)
            .zip(ga_results)
            .map(|((det, emb), ga)| FaceAnalysis {
                detection: det,
                embedding: emb,
                gender: ga.gender,
                age: ga.age,
            })
            .collect();

        Ok(final_results)
    }
}
