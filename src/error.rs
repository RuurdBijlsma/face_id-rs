#[cfg(feature = "hf-hub")]
use hf_hub::api::tokio::ApiError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FaceIdError {
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Image Error: {0}")]
    Image(#[from] image::ImageError),
    #[error("ONNX Runtime Error: {0}")]
    FailedToGetMutableSlice(String),
    #[error("Failed to get mutable slice: {0}")]
    Ort(String),
    #[error("Mutex Poisoned Error: {0}")]
    MutexPoisoned(String),
    #[error("NdArray Error: {0}")]
    NdArray(#[from] ndarray::ShapeError),
    #[error("Image Decoding Error")]
    Decode,
    #[error("Invalid Model: {0}")]
    InvalidModel(String),
    #[cfg(feature = "hf-hub")]
    #[error("Hugging Face Hub error: {0}")]
    HfHub(String),
    #[error("Clustering Error: {0}")]
    Clustering(String),
}

impl<T> From<ort::Error<T>> for FaceIdError {
    fn from(err: ort::Error<T>) -> Self {
        Self::Ort(err.to_string())
    }
}

#[cfg(feature = "hf-hub")]
impl From<ApiError> for FaceIdError {
    fn from(value: ApiError) -> Self {
        Self::HfHub(value.to_string())
    }
}
