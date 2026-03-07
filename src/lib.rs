#![allow(
    clippy::missing_errors_doc,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

pub mod analyzer;
pub mod detector;
pub mod error;
pub mod face_align;
pub mod gender_age;
pub mod model_manager;
pub mod embedder;
