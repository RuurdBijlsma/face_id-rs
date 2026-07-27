use crate::error::FaceIdError;
#[cfg(feature = "hf-hub")]
use hf_hub::{HFClient, split_id};
use std::path::{Path, PathBuf};

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
pub async fn get_hf_model(
    model: HfModel,
    cache_dir: Option<&Path>,
) -> Result<PathBuf, FaceIdError> {
    let client = match cache_dir {
        Some(dir) => HFClient::builder().cache_dir(dir).build()?,
        None => HFClient::new()?,
    };

    let (owner, name) = split_id(&model.id);
    let repo = client.model(owner, name);

    tracing::info!("Fetching {}", &model.file);
    Ok(repo.download_file().filename(&model.file).send().await?)
}
