use crate::error::FaceIdError;
use hf_hub::api::tokio::Api;
use std::path::PathBuf;

#[cfg(feature = "hf-hub")]
pub async fn get_hf_model(model_id: &str, filename: &str) -> Result<PathBuf, FaceIdError> {
    let api = Api::new()?;
    let repo = api.model(model_id.to_owned());

    Ok(repo.get(filename).await?)
}
