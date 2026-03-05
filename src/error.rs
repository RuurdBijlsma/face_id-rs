use thiserror::Error;

#[derive(Error, Debug)]
pub enum DetectorError {
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Image Error: {0}")]
    Image(#[from] image::ImageError),
    #[error("ONNX Runtime Error: {0}")]
    Ort(#[from] ort::Error),
    #[error("NdArray Error: {0}")]
    NdArray(#[from] ndarray::ShapeError),
    #[error("Image Decoding Error")]
    Decode,
    #[error("Invalid Model: {0}")]
    InvalidModel(String),
}