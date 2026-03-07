use crate::detector::{DetectedFace, ScrfdDetector};
use crate::error::FaceIdError;
use crate::face_align::norm_crop;
use crate::gender_age::{GenderAge, GenderAgeEstimator};
use crate::model_manager::HfModel;
use crate::recognizer::ArcFaceEmbedder;
use bon::bon;
use image::DynamicImage;
use ort::ep::ExecutionProviderDispatch;
use std::path::Path;
use std::sync::Mutex;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FaceAnalysis {
    pub detection: DetectedFace,
    pub embedding: Option<Vec<f32>>,
    pub gender_age: Option<GenderAge>,
}

/// Performs detection, alignment, recognition, and gender/age estimation.
pub struct FaceAnalyzer {
    pub detector: Mutex<ScrfdDetector>,
    pub recognizer: Mutex<ArcFaceEmbedder>,
    pub gender_age: Mutex<GenderAgeEstimator>,
}

#[bon]
impl FaceAnalyzer {
    #[builder(finish_fn = build)]
    pub async fn from_hf(
        #[builder(default = HfModel::default_detector())] detector_model: HfModel,
        #[builder(default = HfModel::default_embedder())] embedder_model: HfModel,
        #[builder(default = HfModel::default_gender_age())] gender_age_model: HfModel,
        #[builder(default = &[])] with_execution_providers: &[ExecutionProviderDispatch],
    ) -> Result<Self, FaceIdError> {
        let detector = ScrfdDetector::from_hf()
            .model(detector_model)
            .with_execution_providers(with_execution_providers)
            .build()
            .await?;
        let recognizer = ArcFaceEmbedder::from_hf()
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
            recognizer: Mutex::new(recognizer),
            gender_age: Mutex::new(gender_age),
        })
    }

    /// Creates a new analyzer using local paths to ONNX model files.
    #[builder(finish_fn = build)]
    pub fn new(
        #[builder(start_fn)] det_model: impl AsRef<Path>,
        #[builder(start_fn)] rec_model: impl AsRef<Path>,
        #[builder(start_fn)] attr_model: impl AsRef<Path>,
        #[builder(default = &[])] with_execution_providers: &[ExecutionProviderDispatch],
    ) -> Result<Self, FaceIdError> {
        let detector = ScrfdDetector::builder(det_model)
            .with_execution_providers(with_execution_providers)
            .build()?;

        let recognizer = ArcFaceEmbedder::builder(rec_model)
            .with_execution_providers(with_execution_providers)
            .build()?;

        let gender_age = GenderAgeEstimator::builder(attr_model)
            .with_execution_providers(with_execution_providers)
            .build()?;

        Ok(Self {
            detector: Mutex::new(detector),
            recognizer: Mutex::new(recognizer),
            gender_age: Mutex::new(gender_age),
        })
    }

    /// Performs the full pipeline: detection -> alignment -> embedding -> gender/age estimation.
    pub fn analyze(&self, img: &DynamicImage) -> Result<Vec<FaceAnalysis>, FaceIdError> {
        let detections = {
            let mut det = self
                .detector
                .lock()
                .map_err(|_| FaceIdError::MutexPoisoned("Detector lock poisoned".into()))?;
            det.detect(img)?
        };
        if detections.is_empty() {
            return Ok(vec![]);
        }
        let mut embedder_aligned_crops = Vec::new();
        let mut embedder_indices = Vec::new();
        let mut gender_age_crops = Vec::new();
        let mut gender_age_indices = Vec::new();

        // 1. Prepare crops for both models
        for (idx, detection) in detections.iter().enumerate() {
            // Recognition Alignment
            if let Some(landmarks) = &detection.landmarks {
                if landmarks.len() == 5 {
                    let lms_array: [(f32, f32); 5] = [
                        landmarks[0],
                        landmarks[1],
                        landmarks[2],
                        landmarks[3],
                        landmarks[4],
                    ];
                    let aligned = norm_crop(img, &lms_array, 112);
                    embedder_aligned_crops.push(aligned);
                    embedder_indices.push(idx);
                }
            }

            // Gender/Age Alignment
            let gender_age_crop = GenderAgeEstimator::align_crop(img, &detection.bbox, 96);
            gender_age_crops.push(gender_age_crop);
            gender_age_indices.push(idx);
        }

        let mut results: Vec<FaceAnalysis> = detections
            .into_iter()
            .map(|det| FaceAnalysis {
                detection: det,
                embedding: None,
                gender_age: None,
            })
            .collect();

        // 2. Batch Inference: Recognizer
        if !embedder_aligned_crops.is_empty() {
            let mut embedder = self
                .recognizer
                .lock()
                .map_err(|_| FaceIdError::MutexPoisoned("Recognizer lock poisoned".into()))?;
            let embeddings = embedder.compute_embeddings_batch(&embedder_aligned_crops)?;
            for (batch_idx, original_idx) in embedder_indices.into_iter().enumerate() {
                if let Some(emb) = embeddings.get(batch_idx) {
                    results[original_idx].embedding = Some(emb.clone());
                }
            }
        }

        if !gender_age_crops.is_empty() {
            let mut estimator = self
                .gender_age
                .lock()
                .map_err(|_| FaceIdError::MutexPoisoned("GenderAge lock poisoned".into()))?;
            let ga_results = estimator.estimate_batch(&gender_age_crops)?;
            for (batch_idx, original_idx) in gender_age_indices.into_iter().enumerate() {
                if let Some(res_val) = ga_results.get(batch_idx) {
                    results[original_idx].gender_age = Some(res_val.clone());
                }
            }
        }

        Ok(results)
    }
}
