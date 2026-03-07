#![allow(
    clippy::missing_errors_doc,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

pub mod detector;
pub mod error;
mod model_manager;
pub mod face_align;
pub mod recognizer;
pub mod gender_age;
