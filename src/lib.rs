//! # Face ID
//!
//! `face_id` is a crate for face detection, facial recognition (embeddings),
//! and attribute estimation (age/gender) using ONNX Runtime.
//!
//! ![Face ID](https://raw.githubusercontent.com/RuurdBijlsma/face_id-rs/main/.github/readme_img.png)
//!
//! ## Overview
//!
//! This crate provides a pipeline for facial analysis. It wraps several
//! models (SCRFD for detection, `ArcFace` for recognition) and handles the
//! maths of face alignment and image preprocessing internally.
//!
//! ### The Pipeline
//! 1. **Detection**: Finds bounding boxes and 5-point facial landmarks (eyes, nose, mouth).
//! 2. **Alignment**: Uses the Umeyama algorithm to warp the face into a canonical 112x112 pose.
//! 3. **Analysis**: Runs the aligned crops through specialized models to produce:
//!    - **Embeddings**: 512-dimensional vectors representing identity.
//!    - **Attributes**: Gender and age estimation.
//!
//! ## Quick Start
//!
//! The [`analyzer::FaceAnalyzer`] is the main entry point. It manages the sub-models
//! and performs batch inference for efficiency.
//!
//! ```rust
//! use face_id::analyzer::FaceAnalyzer;
//!
//! #[tokio::main]
//! async fn main() -> color_eyre::Result<()> {
//!     // Initialize the analyzer.
//!     // This downloads default models from HuggingFace on the first run.
//!     let analyzer = FaceAnalyzer::from_hf().build().await?;
//!
//!     let img = image::open("assets/img/crowd.jpg")?;
//!     let faces = analyzer.analyze(&img)?;
//!
//!     for (i, face) in faces.iter().enumerate() {
//!         println!("Face {i}");
//!         println!("    Box: {:?}", &face.detection.bbox);
//!         println!("    Score: {:?}", &face.detection.score); // Confidence score of detection
//!         println!("    Landmarks: {:?}", &face.detection.landmarks); // location of eyes, mouth, nose
//!         
//!         if let Some(ga) = &face.gender_age {
//!             println!("    Gender: {:?}", ga.gender);
//!             println!("    Age: {:?}", ga.age);
//!         }
//!         
//!         if let Some(x) = &face.embedding {
//!             println!("    Embedding [..5]: {:?}", &x[..5]);
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ## Individual Components
//!
//! For more granular control, you can use the individual models directly.
//!
//! ### Detection
//! Use [`detector::ScrfdDetector`] to find faces, their bounding boxes, and eyes/nose/mouth location.
//!
//! ```rust
//! # use face_id::detector::ScrfdDetector;
//! # async fn run() -> color_eyre::Result<()> {
//! let mut detector = ScrfdDetector::from_hf().build().await?;
//! let detections = detector.detect(&image::open("input.jpg")?)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Recognition and Alignment
//! Recognition models like `ArcFace` require faces to be "aligned"—rotated and scaled so that
//! landmarks are in specific positions.
//!
//! ```rust
//! # use face_id::embedder::ArcFaceEmbedder;
//! # use face_id::face_align::norm_crop;
//! # async fn run(img: image::DynamicImage, landmarks: [(f32, f32); 5]) -> color_eyre::Result<()> {
//! let mut embedder = ArcFaceEmbedder::from_hf().build().await?;
//!
//! // Align face using 5-point landmarks (from a detector)
//! let aligned_crop = norm_crop(&img.to_rgb8(), &landmarks, 112);
//!
//! // Generate identity embedding
//! let embedding = embedder.compute_embedding(&aligned_crop)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Loading Local Models
//!
//! If you want to use local ONNX model files instead of downloading from `HuggingFace`,
//! use the builder method.
//!
//! ```rust,no_run
//! use face_id::analyzer::FaceAnalyzer;
//!
//! fn main() -> color_eyre::Result<()> {
//!     let analyzer = FaceAnalyzer::builder(
//!         "models/det.onnx", // Detector
//!         "models/rec.onnx", // Embedder (Recognition)
//!         "models/attr.onnx" // Gender/Age
//!     )
//!     .build()?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Customizing Hugging Face Models
//!
//! You can mix and match specific model versions from Hugging Face repositories.
//! For example, using the medium-complexity `10g_bnkps` detector instead of the default:
//!
//! ```rust
//! use face_id::analyzer::FaceAnalyzer;
//! use face_id::model_manager::HfModel;
//!
//! #[tokio::main]
//! async fn main() -> color_eyre::Result<()> {
//!     let analyzer = FaceAnalyzer::from_hf()
//!         // Specify a smaller detector model than the default:
//!         // > `embedder_model` and `gender_age_model` can also be specified in the builder.
//!         .detector_model(HfModel {
//!             id: "public-data/insightface".to_string(),
//!             file: "models/buffalo_l/det_10g.onnx".to_string(),
//!         })
//!         .detector_input_size((640, 640))
//!         .detector_score_threshold(0.5)
//!         .detector_iou_threshold(0.4)
//!         .build()
//!         .await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Hardware Acceleration
//!
//! This crate supports a variety of Execution Providers (EPs) via `ort`. To use a specific GPU
//! backend, enable the corresponding feature in your `Cargo.toml` (e.g., `cuda`, `tensorrt`, `coreml`).
//!
//! ```rust
//! use face_id::analyzer::FaceAnalyzer;
//! use ort::ep::{DirectML, TensorRT, CUDA, CoreML};
//!
//! # async fn run() -> color_eyre::Result<()> {
//! let analyzer = FaceAnalyzer::from_hf()
//!     .with_execution_providers(&[
//!         CoreML::default().build(), // This will try CoreML, then DirectML, then TensorRT, then CUDA, then CPU.
//!         DirectML::default().build(), // To check if an execution provider if working for you, add `.error_on_failure()`.
//!         TensorRT::default().build(), // Otherwise it will silently try the next execution provider in the list.
//!         CUDA::default().build().error_on_failure(),
//!     ])
//!     .build()
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Features
//!
//! - `hf-hub` (Default): Allows downloading models from Hugging Face.
//! - `copy-dylibs` / `download-binaries` (Default): Simplifies `ort` setup.
//! - `serde`: Enables serialization/deserialization for results.
//! - **Execution Providers**: `cuda`, `tensorrt`, `coreml`, `directml`, `openvino`, etc.

#![allow(
    clippy::missing_errors_doc,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

pub mod analyzer;
pub mod detector;
pub mod embedder;
pub mod error;
pub mod face_align;
pub mod gender_age;
pub mod helpers;
pub mod model_manager;
