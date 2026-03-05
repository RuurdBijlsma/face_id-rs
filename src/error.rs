use thiserror::Error;

#[derive(Error, Debug)]
pub enum DetectorError {
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("OpenCV Error: {0}")]
    OpenCv(#[from] opencv::Error),
    #[error("ONNX Runtime Error: {0}")]
    Ort(#[from] ort::Error),
    #[error("NdArray Error: {0}")]
    NdArray(#[from] ndarray::ShapeError),
    #[error("Image Decoding Error")]
    Decode,
}