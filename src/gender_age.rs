#![allow(clippy::similar_names)]
use crate::detector::BoundingBox;
use crate::error::FaceIdError;
use crate::model_manager::{HfModel, get_hf_model};
use bon::bon;
use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::Array4;
use ort::ep::ExecutionProviderDispatch;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Gender {
    Female = 0,
    Male = 1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GenderAge {
    pub gender: Gender,
    pub age: u8,
}

pub struct GenderAgeEstimator {
    pub session: Session,
    pub input_name: String,
}

#[bon]
impl GenderAgeEstimator {
    #[builder(finish_fn = build)]
    pub async fn from_hf(
        #[builder(default = HfModel::default_gender_age())] model: HfModel,
        #[builder(default = &[])] with_execution_providers: &[ExecutionProviderDispatch],
    ) -> Result<Self, FaceIdError> {
        let model_path = get_hf_model(model).await?;
        Self::builder(model_path)
            .with_execution_providers(with_execution_providers)
            .build()
    }

    #[builder]
    pub fn new(
        #[builder(start_fn)] model_path: impl AsRef<Path>,
        #[builder(default = &[])] with_execution_providers: &[ExecutionProviderDispatch],
    ) -> Result<Self, FaceIdError> {
        let session = Session::builder()?
            .with_execution_providers(with_execution_providers)?
            .commit_from_file(model_path)?;

        let input_name = session.inputs()[0].name().to_string();

        Ok(Self {
            session,
            input_name,
        })
    }

    /// Estimates gender and age for a batch of cropped face images.
    pub fn estimate_batch(
        &mut self,
        face_imgs: &[ImageBuffer<Rgb<u8>, Vec<u8>>],
    ) -> Result<Vec<GenderAge>, FaceIdError> {
        if face_imgs.is_empty() {
            return Ok(vec![]);
        }

        let input_tensor = Self::create_input_tensor_batch(face_imgs)?;
        let input_value = Value::from_array(input_tensor)?;
        let outputs = self
            .session
            .run(ort::inputs![&self.input_name => input_value])?;

        let output_tensor = outputs[0].try_extract_array::<f32>()?;
        let batch_size = face_imgs.len();

        if output_tensor.ndim() != 2 || output_tensor.shape()[0] != batch_size || output_tensor.shape()[1] != 3 {
             return Err(FaceIdError::Ort(format!(
                "GenderAge output shape mismatch: expected [{batch_size}, 3], got {:?}",
                output_tensor.shape()
            )));
        }

        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let prob_female = output_tensor[[i, 0]];
            let prob_male = output_tensor[[i, 1]];
            let age_raw = output_tensor[[i, 2]];

            let gender = if prob_male > prob_female {
                Gender::Male
            } else {
                Gender::Female
            };
            let age = (age_raw * 100.0).round().clamp(0.0, 100.0) as u8;
            results.push(GenderAge { gender, age });
        }

        Ok(results)
    }

    /// Estimates gender and age from an image and a detected face bounding box.
    pub fn estimate(
        &mut self,
        img: &DynamicImage,
        bbox: &BoundingBox,
    ) -> Result<GenderAge, FaceIdError> {
        let rgb_img = img.to_rgb8();
        let cropped_face = Self::align_crop(&rgb_img, bbox, 96);
        let results = self.estimate_batch(&[cropped_face])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| FaceIdError::Ort("GenderAge failed to produce output".into()))
    }

    /// `InsightFace` Attribute alignment: Creates a square crop based on the `BBox`
    /// with a 1.5x expansion factor to include context.
    #[must_use]
    pub fn align_crop(
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        bbox: &BoundingBox,
        output_size: u32,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let (img_w, img_h) = img.dimensions();
        let bbox = bbox.scale(img_w, img_h);

        let w = bbox.width();
        let h = bbox.height();
        let cx = bbox.x1 + w / 2.0;
        let cy = bbox.y1 + h / 2.0;

        // Use 1.5x the largest dimension to create a square crop
        let side = w.max(h) * 1.5;

        let x1 = (cx - side / 2.0) as i32;
        let y1 = (cy - side / 2.0) as i32;
        let side_u = side as u32;

        // We use a canvas to handle out-of-bounds crops (padding with black)
        let mut canvas = ImageBuffer::new(side_u, side_u);

        // Calculate the overlap between the desired crop and the actual image
        let src_x = x1.max(0) as u32;
        let src_y = y1.max(0) as u32;
        let src_x2 = (x1 + side_u.cast_signed()).min(img_w.cast_signed()) as u32;
        let src_y2 = (y1 + side_u.cast_signed()).min(img_h.cast_signed()) as u32;

        if src_x2 > src_x && src_y2 > src_y {
            let width = src_x2 - src_x;
            let height = src_y2 - src_y;

            // Manual copy from buffer to canvas
            for y in 0..height {
                for x in 0..width {
                    let pixel = img.get_pixel(src_x + x, src_y + y);
                    let dst_x = (src_x.cast_signed() - x1) as u32 + x;
                    let dst_y = (src_y.cast_signed() - y1) as u32 + y;
                    canvas.put_pixel(dst_x, dst_y, *pixel);
                }
            }
        }

        // Resize the padded square crop to the model input size (96x96)
        DynamicImage::ImageRgb8(canvas)
            .resize_exact(
                output_size,
                output_size,
                image::imageops::FilterType::Triangle,
            )
            .to_rgb8()
    }

    fn create_input_tensor_batch(
        imgs: &[ImageBuffer<Rgb<u8>, Vec<u8>>],
    ) -> Result<Array4<f32>, FaceIdError> {
        let batch_size = imgs.len();
        let mut array = Array4::<f32>::zeros((batch_size, 3, 96, 96));

        let data = array.as_slice_memory_order_mut().ok_or_else(|| {
            FaceIdError::FailedToGetMutableSlice("Failed to get mutable slice from array".into())
        })?;

        let channel_stride = 96 * 96;
        for (batch_idx, img) in imgs.iter().enumerate() {
            let (w, h) = img.dimensions();
            if w != 96 || h != 96 {
                return Err(FaceIdError::InvalidModel(format!(
                    "GenderAge requires 96x96 input, got {w}x{h}"
                )));
            }

            let raw = img.as_raw();
            let batch_offset = batch_idx * 3 * channel_stride;

            // Buffalo-L attribute genderage.onnx expects: BGR channel order, 0-255 range
            for (i, chunk) in raw.chunks_exact(3).enumerate() {
                data[batch_offset + i] = f32::from(chunk[2]); // B
                data[batch_offset + i + channel_stride] = f32::from(chunk[1]); // G
                data[batch_offset + i + 2 * channel_stride] = f32::from(chunk[0]); // R
            }
        }

        Ok(array)
    }

    pub fn create_input_tensor(
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<Array4<f32>, FaceIdError> {
        Self::create_input_tensor_batch(std::slice::from_ref(img))
    }
}
