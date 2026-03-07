use crate::detector::BoundingBox;
use crate::error::FaceIdError;
use crate::model_manager::{HfModel, get_hf_model};
use bon::bon;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use ndarray::{Array4, s};
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

#[derive(Debug, Clone)]
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

        let input_tensor = self.create_input_tensor_batch(face_imgs)?;
        let input_value = Value::from_array(input_tensor)?;
        let outputs = self
            .session
            .run(ort::inputs![&self.input_name => input_value])?;

        let output_tensor = outputs[0].try_extract_array::<f32>()?;

        // Handle output shape [N, 3]
        let batch_size = face_imgs.len();
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
            let age = (age_raw * 100.0).round().max(0.0).min(100.0) as u8;
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
        let cropped_face = Self::align_crop(img, bbox, 96);
        let results = self.estimate_batch(&[cropped_face])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| FaceIdError::Ort("GenderAge failed to produce output".into()))
    }

    /// InsightFace Attribute alignment: Creates a square crop based on the BBox
    /// with a 1.5x expansion factor to include context.
    pub fn align_crop(
        img: &DynamicImage,
        bbox: &BoundingBox,
        output_size: u32,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let (img_w, img_h) = img.dimensions();

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
        let src_x2 = (x1 + side_u as i32).min(img_w as i32) as u32;
        let src_y2 = (y1 + side_u as i32).min(img_h as i32) as u32;

        if src_x2 > src_x && src_y2 > src_y {
            let width = src_x2 - src_x;
            let height = src_y2 - src_y;

            let sub_img = img.view(src_x, src_y, width, height);

            // Where to paste in the canvas
            let dst_x = (src_x as i32 - x1) as u32;
            let dst_y = (src_y as i32 - y1) as u32;

            image::imageops::overlay(&mut canvas, &sub_img.to_image(), dst_x as i64, dst_y as i64);
        }

        // Resize the padded square crop to the model input size (96x96)
        let dynamic_canvas = DynamicImage::ImageRgba8(canvas);
        dynamic_canvas
            .resize_exact(
                output_size,
                output_size,
                image::imageops::FilterType::Triangle,
            )
            .to_rgb8()
    }

    fn create_input_tensor_batch(
        &self,
        imgs: &[ImageBuffer<Rgb<u8>, Vec<u8>>],
    ) -> Result<Array4<f32>, FaceIdError> {
        let batch_size = imgs.len();
        let mut array = Array4::<f32>::zeros((batch_size, 3, 96, 96));

        for (batch_idx, img) in imgs.iter().enumerate() {
            let (w, h) = img.dimensions();
            if w != 96 || h != 96 {
                return Err(FaceIdError::InvalidModel(format!(
                    "GenderAge requires 96x96 input, got {}x{}",
                    w, h
                )));
            }

            let raw = img.as_raw();
            let mut view = array.slice_mut(s![batch_idx, .., .., ..]);

            // Buffalo-L attribute genderage.onnx expects: BGR channel order, 0-255 range
            let (b_plane, rest) = view
                .as_slice_memory_order_mut()
                .ok_or_else(|| FaceIdError::Ort("Failed to get mutable slice".into()))?
                .split_at_mut(96 * 96);
            let (g_plane, r_plane) = rest.split_at_mut(96 * 96);

            for (i, pixel) in raw.chunks_exact(3).enumerate() {
                r_plane[i] = f32::from(pixel[0]); // R
                g_plane[i] = f32::from(pixel[1]); // G
                b_plane[i] = f32::from(pixel[2]); // B
            }
        }

        Ok(array)
    }

    pub fn create_input_tensor(
        &self,
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<Array4<f32>, FaceIdError> {
        self.create_input_tensor_batch(std::slice::from_ref(img))
    }
}
