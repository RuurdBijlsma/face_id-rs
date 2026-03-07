use crate::detector::BoundingBox;
use crate::error::FaceIdError;
use crate::model_manager::{get_hf_model, HfModel};
use bon::bon;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
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

    /// Estimates gender and age from an image and a detected face bounding box.
    pub fn estimate(
        &mut self,
        img: &DynamicImage,
        bbox: &BoundingBox,
    ) -> Result<GenderAge, FaceIdError> {
        // 1. Align and Crop (Square 96x96)
        let cropped_face = self.align_crop(img, bbox, 96);

        // 2. Preprocess (NCHW + Normalization)
        let input_tensor = self.create_input_tensor(&cropped_face)?;
        let input_value = Value::from_array(input_tensor)?;

        // 3. Run Inference
        let outputs = self
            .session
            .run(ort::inputs![&self.input_name => input_value])?;
        let output_tensor = outputs[0].try_extract_array::<f32>()?;

        // 4. Post-process
        // Output is typically [1, 3] -> [prob_female, prob_male, age_scaled]
        let prob_female = output_tensor[[0, 0]];
        let prob_male = output_tensor[[0, 1]];
        let age_raw = output_tensor[[0, 2]];

        let gender = if prob_male > prob_female {
            Gender::Male
        } else {
            Gender::Female
        };

        // Age is regressed and needs to be multiplied by 100
        let age = (age_raw * 100.0).round().max(0.0).min(100.0) as u8;

        Ok(GenderAge { gender, age })
    }

    /// InsightFace Attribute alignment: Creates a square crop based on the BBox
    /// with a 1.5x expansion factor to include context.
    fn align_crop(
        &self,
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

    fn create_input_tensor(
        &self,
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<Array4<f32>, FaceIdError> {
        let (w, h) = img.dimensions();
        let mut array = Array4::<f32>::zeros((1, 3, h as usize, w as usize));
        let raw = img.as_raw();

        // Buffalo-L attribute genderage.onnx expects:
        // 1. BGR channel order
        // 2. Raw pixel values (0.0 to 255.0), no normalization/mean subtraction

        let (b_plane, rest) = array
            .as_slice_memory_order_mut()
            .expect("Contiguous array")
            .split_at_mut((w * h) as usize);
        let (g_plane, r_plane) = rest.split_at_mut((w * h) as usize);

        for (i, pixel) in raw.chunks_exact(3).enumerate() {
            r_plane[i] = f32::from(pixel[0]); // R
            g_plane[i] = f32::from(pixel[1]); // G
            b_plane[i] = f32::from(pixel[2]); // B
        }

        Ok(array)
    }
}
