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

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FaceAnalysis {
    pub detection: DetectedFace,
    pub embedding: Option<Vec<f32>>,
    pub gender_age: Option<GenderAge>,
}

pub struct FaceAnalyzer {
    pub detector: ScrfdDetector,
    pub recognizer: ArcFaceEmbedder,
    pub gender_age: GenderAgeEstimator,
}

#[bon]
impl FaceAnalyzer {
    /// Creates a new analyzer by downloading/fetching default models from HuggingFace.
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
            detector,
            recognizer,
            gender_age,
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
            detector,
            recognizer,
            gender_age,
        })
    }

    /// Performs the full pipeline: detection -> alignment -> embedding -> gender/age estimation.
    pub fn analyze(&mut self, img: &DynamicImage) -> Result<Vec<FaceAnalysis>, FaceIdError> {
        // 1. Detect faces
        let faces = self.detector.detect(img)?;
        let mut results = Vec::with_capacity(faces.len());

        for face in faces {
            // 2. Alignment & Embedding (requires landmarks)
            let mut embedding = None;
            if let Some(landmarks) = &face.landmarks {
                if landmarks.len() == 5 {
                    let lms_array: [(f32, f32); 5] = [
                        landmarks[0],
                        landmarks[1],
                        landmarks[2],
                        landmarks[3],
                        landmarks[4],
                    ];
                    // Standard ArcFace 112x112 alignment
                    let aligned = norm_crop(img, &lms_array, 112);
                    embedding = Some(self.recognizer.compute_embedding(&aligned)?);
                }
            }

            // 3. Gender and Age estimation (uses original image and bbox)
            let gender_age = self.gender_age.estimate(img, &face.bbox).ok();

            results.push(FaceAnalysis {
                detection: face,
                embedding,
                gender_age,
            });
        }

        Ok(results)
    }
}
