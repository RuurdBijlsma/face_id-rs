use thiserror::Error;

#[derive(Error, Debug)]
pub enum DetectorError {
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Image Error: {0}")]
    Image(#[from] image::ImageError),
    #[error("ONNX Runtime Error: {0}")]
    Ort(String),
    #[error("NdArray Error: {0}")]
    NdArray(#[from] ndarray::ShapeError),
    #[error("Image Decoding Error")]
    Decode,
    #[error("Invalid Model: {0}")]
    InvalidModel(String),
}

impl<T> From<ort::Error<T>> for DetectorError {
    fn from(err: ort::Error<T>) -> Self {
        Self::Ort(err.to_string())
    }
}
