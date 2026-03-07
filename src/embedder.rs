use crate::error::FaceIdError;
use crate::model_manager::{HfModel, get_hf_model};
use bon::bon;
use image::{ImageBuffer, Rgb};
use ndarray::{Array2, Array4, Axis, s};
use ort::ep::ExecutionProviderDispatch;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;

pub struct ArcFaceEmbedder {
    pub session: Session,
    pub input_name: String,
}

#[bon]
impl ArcFaceEmbedder {
    #[builder(finish_fn = build)]
    pub async fn from_hf(
        #[builder(default = HfModel::default_embedder())] model: HfModel,
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

    pub fn compute_embeddings_batch(
        &mut self,
        aligned_imgs: &[ImageBuffer<Rgb<u8>, Vec<u8>>],
    ) -> Result<Vec<Vec<f32>>, FaceIdError> {
        if aligned_imgs.is_empty() {
            return Ok(vec![]);
        }

        let input_tensor = Self::create_input_tensor_batch(aligned_imgs)?;
        let input_value = Value::from_array(input_tensor)?;

        let outputs = self
            .session
            .run(ort::inputs![&self.input_name => input_value])?;

        let mut output_tensor = outputs[0]
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality::<ndarray::Ix2>()?;
        Self::l2_normalize_batch(&mut output_tensor);

        let batch_size = output_tensor.shape()[0];
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            results.push(output_tensor.slice(s![i, ..]).to_vec());
        }

        Ok(results)
    }

    fn create_input_tensor_batch(
        imgs: &[ImageBuffer<Rgb<u8>, Vec<u8>>],
    ) -> Result<Array4<f32>, FaceIdError> {
        let batch_size = imgs.len();
        // Shape: [N, 3, 112, 112]
        let mut array = Array4::<f32>::zeros((batch_size, 3, 112, 112));

        let data = array.as_slice_memory_order_mut().ok_or_else(|| {
            FaceIdError::Ort("Failed to get mutable slice".into())
        })?;

        let channel_stride = 112 * 112;
        for (batch_idx, img) in imgs.iter().enumerate() {
            let (w, h) = img.dimensions();
            if w != 112 || h != 112 {
                return Err(FaceIdError::InvalidModel(format!(
                    "ArcFace requires 112x112 input, got {w}x{h}"
                )));
            }

            let raw = img.as_raw();
            let batch_offset = batch_idx * 3 * channel_stride;

            for (i, chunk) in raw.chunks_exact(3).enumerate() {
                data[batch_offset + i] = (f32::from(chunk[0]) - 127.5) / 127.5;
                data[batch_offset + i + channel_stride] = (f32::from(chunk[1]) - 127.5) / 127.5;
                data[batch_offset + i + 2 * channel_stride] = (f32::from(chunk[2]) - 127.5) / 127.5;
            }
        }

        Ok(array)
    }

    /// Takes an ALIGNED face image (112x112) and returns a normalized 512-d embedding.
    pub fn compute_embedding(
        &mut self,
        aligned_img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<Vec<f32>, FaceIdError> {
        let mut results = self.compute_embeddings_batch(std::slice::from_ref(aligned_img))?;
        results
            .pop()
            .ok_or_else(|| FaceIdError::Ort("Embedder failed to produce an embedding".into()))
    }

    /// Preprocessing wrapper for a single image.
    pub fn create_input_tensor(
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<Array4<f32>, FaceIdError> {
        Self::create_input_tensor_batch(std::slice::from_ref(img))
    }

    /// Performs vectorized L2 normalization on a 2D array of embeddings [N, Dim] in-place.
    pub fn l2_normalize_batch(embeddings: &mut Array2<f32>) {
        let view = embeddings.view();
        let sq_sums = (&view * &view).sum_axis(Axis(1));
        let inv_norms = sq_sums
            .mapv(|x| 1.0 / x.max(1e-12).sqrt())
            .insert_axis(Axis(1));
        *embeddings *= &inv_norms;
    }

    /// Normalizes a single vector.
    pub fn l2_normalize(vec: &mut [f32]) {
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            let inv_norm = 1.0 / norm;
            for x in vec.iter_mut() {
                *x *= inv_norm;
            }
        }
    }

    /// Computes cosine similarity between two L2-normalized embeddings.
    /// Range: -1.0 to 1.0 (Higher is more similar).
    #[must_use]
    pub fn compute_similarity(emb1: &[f32], emb2: &[f32]) -> f32 {
        emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum()
    }
}
