use crate::error::FaceIdError;
use crate::model_manager::{get_hf_model, HfModel};
use bon::bon;
use image::{ImageBuffer, Rgb};
use ndarray::Array4;
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
        dbg!(&session);

        let input_name = session.inputs()[0].name().to_string();

        Ok(Self {
            session,
            input_name,
        })
    }

    /// Takes an ALIGNED face image (112x112) and returns a normalized 512-d embedding.
    pub fn compute_embedding(
        &mut self,
        aligned_img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<Vec<f32>, FaceIdError> {
        let input_tensor = self.create_input_tensor(aligned_img)?;
        let input_value = Value::from_array(input_tensor)?;

        // Fixed: removed the '?' from inside the run() call
        let outputs = self
            .session
            .run(ort::inputs![&self.input_name => input_value])?;

        // Extract the first output (the embedding)
        let output_tensor = outputs[0].try_extract_array::<f32>()?;

        // Flatten to Vec and normalize
        let mut embedding: Vec<f32> = output_tensor.iter().copied().collect();
        Self::l2_normalize(&mut embedding);

        Ok(embedding)
    }

    /// Optimized preprocessing: Converts HWC ImageBuffer to NCHW Array4
    /// and applies ArcFace normalization: (pixel - 127.5) / 127.5
    pub fn create_input_tensor(
        &self,
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<Array4<f32>, FaceIdError> {
        let (w, h) = img.dimensions();
        if w != 112 || h != 112 {
            return Err(FaceIdError::InvalidModel(format!(
                "ArcFace requires 112x112 input, got {}x{}",
                w, h
            )));
        }

        let mut array = Array4::<f32>::zeros((1, 3, h as usize, w as usize));
        let raw = img.as_raw();

        // Split the array into R, G, and B planes for NCHW layout
        // This allows us to fill the memory contiguously without nested loops
        let (r_plane, rest) = array
            .as_slice_memory_order_mut()
            .expect("Array is contiguous")
            .split_at_mut(112 * 112);
        let (g_plane, b_plane) = rest.split_at_mut(112 * 112);

        for (i, pixel) in raw.chunks_exact(3).enumerate() {
            r_plane[i] = (f32::from(pixel[0]) - 127.5) / 127.5;
            g_plane[i] = (f32::from(pixel[1]) - 127.5) / 127.5;
            b_plane[i] = (f32::from(pixel[2]) - 127.5) / 127.5;
        }

        Ok(array)
    }

    pub fn l2_normalize(vec: &mut [f32]) {
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            let inv_norm = 1.0 / norm;
            for x in vec.iter_mut() {
                *x *= inv_norm;
            }
        }
    }

    /// Computes cosine similarity between two L2-normalized embeddings.
    /// Range: -1.0 to 1.0 (Higher is more similar).
    pub fn compute_similarity(emb1: &[f32], emb2: &[f32]) -> f32 {
        emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum()
    }
}
