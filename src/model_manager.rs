use crate::error::FaceIdError;
use hf_hub::api::tokio::Api;
use std::path::PathBuf;

pub struct HfModel {
    pub id: String,
    pub file: String,
}

impl HfModel {
    #[must_use] 
    pub fn default_detector() -> Self {
        Self {
            id: "RuteNL/SCRFD-face-detection-ONNX".to_owned(),
            file: "34g_gnkps.onnx".to_owned(),
        }
    }

    #[must_use] 
    pub fn default_embedder() -> Self {
        Self {
            id: "public-data/insightface".to_owned(),
            file: "models/buffalo_l/w600k_r50.onnx".to_owned(),
        }
    }

    #[must_use] 
    pub fn default_gender_age() -> Self {
        Self {
            id: "public-data/insightface".to_owned(),
            file: "models/buffalo_l/genderage.onnx".to_owned(),
        }
    }
}

#[cfg(feature = "hf-hub")]
pub async fn get_hf_model(model: HfModel) -> Result<PathBuf, FaceIdError> {
    let api = Api::new()?;
    let repo = api.model(model.id);

    Ok(repo.get(&model.file).await?)
}
