use crate::error::DetectorError;
use bon::bon;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use ndarray::{Array2, Array4, Ix2, s};
use ort::{
    session::{Session, SessionOutputs},
    value::Value,
};
use std::path::Path;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Face {
    pub bbox: BBox,
    pub landmarks: Option<Vec<(f32, f32)>>,
    pub score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    #[must_use]
    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }
    
    #[must_use]
    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }
    
    #[must_use]
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }
}

#[derive(Debug, Clone)]
pub struct PreprocessParams {
    ratio: f32,
    x_offset: f32,
    y_offset: f32,
}

#[derive(Debug, Clone)]
pub struct DetectorConfig {
    pub input_size: (u32, u32),
    pub score_threshold: f32,
    pub iou_threshold: f32,
}

pub struct ScrfdDetector {
    pub session: Session,
    pub config: DetectorConfig,
    pub anchors: Vec<Array2<f32>>,
    pub strides: Vec<i32>,
    pub input_name: String,
}

#[bon]
impl ScrfdDetector {
    #[builder]
    pub fn new(
        #[builder(start_fn)] model_path: impl AsRef<Path>,
        #[builder(default = (640, 640))] input_size: (u32, u32),
        #[builder(default = 0.5)] score_threshold: f32,
        #[builder(default = 0.4)] iou_threshold: f32,
    ) -> Result<Self, DetectorError> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        dbg!(&session);
        let input_name = session.inputs()[0].name().to_string();
        let config = DetectorConfig {
            input_size,
            score_threshold,
            iou_threshold,
        };

        let mut strides: Vec<i32> = session
            .outputs()
            .iter()
            .filter_map(|output| output.name().strip_prefix("score_")?.parse::<i32>().ok())
            .collect();
            
        strides.sort_unstable();

        if strides.is_empty() {
            return Err(DetectorError::InvalidModel("No stride info found".into()));
        }

        // Dynamically determine how many anchors per location are in this model
        let first_stride = strides[0];
        let score_name = format!("score_{first_stride}");
        let score_output = session
            .outputs()
            .iter()
            .find(|o| o.name() == score_name)
            .ok_or_else(|| DetectorError::InvalidModel(format!("Missing output: {score_name}")))?;

        let num_anchors = if let Some(shape) = score_output.dtype().tensor_shape() {
            let h = (config.input_size.1 / first_stride as u32) as i64;
            let w = (config.input_size.0 / first_stride as u32) as i64;

            // Handle [batch, anchors, 1] or [anchors, 1]
            let total_anchors = if shape.len() > 1 {
                shape.iter().rev().nth(1).copied().unwrap_or(0)
            } else {
                shape.iter().next().copied().unwrap_or(0)
            };

            if h * w == 0 {
                2
            } else {
                (total_anchors / (h * w)) as usize
            }
        } else {
            2
        };

        let anchors = strides
            .iter()
            .map(|&s| Self::generate_anchors(config.input_size, s, num_anchors))
            .collect();

        Ok(Self {
            session,
            config,
            anchors,
            strides,
            input_name,
        })
    }

    pub fn detect(&mut self, img: &DynamicImage) -> Result<Vec<Face>, DetectorError> {
        let (processed_img, params) = self.preprocess(img);
        let input_tensor = self.create_input_tensor(&processed_img)?;
        let input_value = Value::from_array(input_tensor)?;
        let inputs = ort::inputs![&self.input_name => input_value];
        let outputs = self.session.run(inputs)?;

        Self::postprocess(
            &outputs,
            &params,
            &self.strides,
            &self.anchors,
            &self.config,
        )
    }

    fn generate_anchors(input_size: (u32, u32), stride: i32, num_anchors: usize) -> Array2<f32> {
        let h = (input_size.1 / stride as u32) as usize;
        let w = (input_size.0 / stride as u32) as usize;
        let stride_f = stride as f32;

        Array2::from_shape_fn((h * w * num_anchors, 2), |(i, j)| {
            let pixel_idx = i / num_anchors;
            let y = (pixel_idx / w) as f32 * stride_f;
            let x = (pixel_idx % w) as f32 * stride_f;
            if j == 0 { x } else { y }
        })
    }

    #[must_use]
    pub fn preprocess(
        &self,
        img: &DynamicImage,
    ) -> (ImageBuffer<Rgb<u8>, Vec<u8>>, PreprocessParams) {
        let (w_in, h_in) = self.config.input_size;
        let (w_orig, h_orig) = img.dimensions();

        let ratio = (w_in as f32 / w_orig as f32).min(h_in as f32 / h_orig as f32);
        let w_new = (w_orig as f32 * ratio).round() as u32;
        let h_new = (h_orig as f32 * ratio).round() as u32;

        let resized = img.resize_exact(w_new, h_new, image::imageops::FilterType::CatmullRom);

        let mut padded = ImageBuffer::new(w_in, h_in);
        let x_offset = (w_in - w_new) as f32 / 2.0;
        let y_offset = (h_in - h_new) as f32 / 2.0;

        image::imageops::overlay(
            &mut padded,
            &resized.to_rgb8(),
            x_offset as i64,
            y_offset as i64,
        );

        (
            padded,
            PreprocessParams {
                ratio,
                x_offset,
                y_offset,
            },
        )
    }

    pub fn create_input_tensor(
        &self,
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<Array4<f32>, DetectorError> {
        let (width, height) = img.dimensions();
        let w = width as usize;
        let h = height as usize;
        let raw = img.as_raw();

        let mut array = Array4::<f32>::zeros((1, 3, h, w));
        
        let (r_plane, rest) = array.as_slice_memory_order_mut()
            .expect("Array was just created contiguously")
            .split_at_mut(h * w);
        let (g_plane, b_plane) = rest.split_at_mut(h * w);

        // Optimize: convert HWC directly to NCHW normalized without intermediate allocations
        // The image is internally contiguous R, G, B triplets.
        for (i, pixel) in raw.chunks_exact(3).enumerate() {
            r_plane[i] = (f32::from(pixel[0]) - 127.5) / 128.0;
            g_plane[i] = (f32::from(pixel[1]) - 127.5) / 128.0;
            b_plane[i] = (f32::from(pixel[2]) - 127.5) / 128.0;
        }

        Ok(array)
    }

    pub fn postprocess(
        outputs: &SessionOutputs,
        params: &PreprocessParams,
        strides: &[i32],
        anchors_list: &[Array2<f32>],
        config: &DetectorConfig,
    ) -> Result<Vec<Face>, DetectorError> {
        let mut candidate_faces = Vec::new();

        for (idx, &stride) in strides.iter().enumerate() {
            let score_key = format!("score_{stride}");
            let bbox_key = format!("bbox_{stride}");
            let kps_key = format!("kps_{stride}");

            let scores = Self::extract_and_reshape(outputs, &score_key)?;
            let bboxes = Self::extract_and_reshape(outputs, &bbox_key)?;
            let kps = Self::extract_and_reshape(outputs, &kps_key)?;

            let anchors = &anchors_list[idx];
            let stride_f = stride as f32;

            for i in 0..scores.nrows() {
                let score = scores[[i, 0]];
                if score < config.score_threshold {
                    continue;
                }

                let dist = bboxes.slice(s![i, ..]);
                let anchor = anchors.slice(s![i, ..]);
                let anchor_x = anchor[0];
                let anchor_y = anchor[1];

                let x1 = (dist[0].mul_add(-stride_f, anchor_x) - params.x_offset) / params.ratio;
                let y1 = (dist[1].mul_add(-stride_f, anchor_y) - params.y_offset) / params.ratio;
                let x2 = (dist[2].mul_add(stride_f, anchor_x) - params.x_offset) / params.ratio;
                let y2 = (dist[3].mul_add(stride_f, anchor_y) - params.y_offset) / params.ratio;

                let kps_dist = kps.slice(s![i, ..]);
                let mut landmarks = Vec::with_capacity(5);
                for j in 0..5 {
                    let lx = (kps_dist[j * 2].mul_add(stride_f, anchor_x) - params.x_offset)
                        / params.ratio;
                    let ly = (kps_dist[j * 2 + 1].mul_add(stride_f, anchor_y) - params.y_offset)
                        / params.ratio;
                    landmarks.push((lx, ly));
                }

                candidate_faces.push(Face {
                    bbox: BBox { x1, y1, x2, y2 },
                    landmarks: Some(landmarks),
                    score,
                });
            }
        }

        Ok(Self::perform_non_maximum_suppression(
            candidate_faces,
            config.iou_threshold,
        ))
    }

    fn perform_non_maximum_suppression(mut faces: Vec<Face>, iou_threshold: f32) -> Vec<Face> {
        // Sort faces by score descending
        faces.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut kept_faces: Vec<Face> = Vec::with_capacity(faces.len());
        for face in faces {
            let is_suppressed = kept_faces.iter().any(|kept| {
                Self::compute_intersection_over_union(&face.bbox, &kept.bbox) > iou_threshold
            });

            if !is_suppressed {
                kept_faces.push(face);
            }
        }

        kept_faces
    }

    fn extract_and_reshape(
        outputs: &SessionOutputs,
        key: &str,
    ) -> Result<Array2<f32>, DetectorError> {
        let array = outputs[key].try_extract_array::<f32>()?;
        if array.ndim() == 3 && array.shape()[0] == 1 {
            Ok(array
                .view()
                .to_shape((array.shape()[1], array.shape()[2]))?
                .to_owned()
                .into_dimensionality::<Ix2>()?)
        } else {
            Ok(array.to_owned().into_dimensionality::<Ix2>()?)
        }
    }

    fn compute_intersection_over_union(a: &BBox, b: &BBox) -> f32 {
        let x1 = a.x1.max(b.x1);
        let y1 = a.y1.max(b.y1);
        let x2 = a.x2.min(b.x2);
        let y2 = a.y2.min(b.y2);
        
        let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        if intersection <= 0.0 {
            return 0.0;
        }
        
        intersection / (a.area() + b.area() - intersection)
    }
}
