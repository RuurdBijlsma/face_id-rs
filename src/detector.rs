use crate::error::DetectorError;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use ndarray::{Array4, Axis, Ix2, s, Array2};
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

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    #[must_use]
    pub fn width(&self) -> f32 { self.x2 - self.x1 }
    #[must_use]
    pub fn height(&self) -> f32 { self.y2 - self.y1 }
    #[must_use]
    pub fn area(&self) -> f32 { self.width() * self.height() }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DetectorConfig {
    pub input_size: (u32, u32),
    pub score_threshold: f32,
    pub iou_threshold: f32,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            input_size: (640, 640),
            score_threshold: 0.5,
            iou_threshold: 0.4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PreprocessParams {
    ratio: f32,
    x_offset: f32,
    y_offset: f32,
}

pub struct ScrfdDetector {
    pub session: Session,
    pub config: DetectorConfig,
    pub anchors: Vec<Array2<f32>>,
    pub strides: Vec<i32>,
    pub input_name: String,
}

impl ScrfdDetector {
    pub fn new(
        model_path: impl AsRef<Path>,
        config: DetectorConfig,
    ) -> Result<Self, DetectorError> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        let input_name = session.inputs()[0].name().to_string();

        let mut strides: Vec<i32> = session
            .outputs()
            .iter()
            .filter_map(|output| {
                if output.name().starts_with("score_") {
                    output.name()["score_".len()..].parse::<i32>().ok()
                } else {
                    None
                }
            })
            .collect();
        strides.sort_unstable();

        if strides.is_empty() {
            return Err(DetectorError::InvalidModel("No stride info found".into()));
        }

        let anchors = strides
            .iter()
            .map(|&s| Self::generate_anchors(config.input_size, s))
            .collect();

        Ok(Self { session, config, anchors, strides, input_name })
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

    fn generate_anchors(input_size: (u32, u32), stride: i32) -> Array2<f32> {
        let h = (input_size.1 / stride as u32) as usize;
        let w = (input_size.0 / stride as u32) as usize;
        let mut anchors = Array2::zeros((h * w * 2, 2));

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 2;
                let val_x = x as f32 * stride as f32;
                let val_y = y as f32 * stride as f32;
                anchors[[idx, 0]] = val_x;
                anchors[[idx, 1]] = val_y;
                anchors[[idx + 1, 0]] = val_x;
                anchors[[idx + 1, 1]] = val_y;
            }
        }
        anchors
    }

    #[must_use]
    pub fn preprocess(&self, img: &DynamicImage) -> (ImageBuffer<Rgb<u8>, Vec<u8>>, PreprocessParams) {
        let (w_in, h_in) = self.config.input_size;
        let (w_orig, h_orig) = img.dimensions();

        let ratio = (w_in as f32 / w_orig as f32).min(h_in as f32 / h_orig as f32);
        let w_new = (w_orig as f32 * ratio).round() as u32;
        let h_new = (h_orig as f32 * ratio).round() as u32;

        let resized = img.resize_exact(w_new, h_new, image::imageops::FilterType::CatmullRom);

        let mut padded = ImageBuffer::new(w_in, h_in);
        let x_offset = (w_in - w_new) as f32 / 2.0;
        let y_offset = (h_in - h_new) as f32 / 2.0;

        image::imageops::overlay(&mut padded, &resized.to_rgb8(), x_offset as i64, y_offset as i64);

        (padded, PreprocessParams { ratio, x_offset, y_offset })
    }

    pub fn create_input_tensor(&self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Array4<f32>, DetectorError> {
        let (width, height) = img.dimensions();

        // Convert ImageBuffer to ndarray (H, W, C)
        let mut array = ndarray::Array3::zeros((height as usize, width as usize, 3));
        for (x, y, pixel) in img.enumerate_pixels() {
            let rgb = pixel.0;
            array[[y as usize, x as usize, 0]] = (f32::from(rgb[0]) - 127.5) / 128.0;
            array[[y as usize, x as usize, 1]] = (f32::from(rgb[1]) - 127.5) / 128.0;
            array[[y as usize, x as usize, 2]] = (f32::from(rgb[2]) - 127.5) / 128.0;
        }

        // HWC to CHW
        let array = array.permuted_axes([2, 0, 1]);
        // Add batch dimension: NCHW
        Ok(array.insert_axis(Axis(0)))
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

            let scores = outputs[score_key.as_str()].try_extract_array::<f32>()?.into_dimensionality::<Ix2>()?;
            let bboxes = outputs[bbox_key.as_str()].try_extract_array::<f32>()?.into_dimensionality::<Ix2>()?;
            let kps = outputs[kps_key.as_str()].try_extract_array::<f32>()?.into_dimensionality::<Ix2>()?;

            let anchors = &anchors_list[idx];

            for i in 0..scores.nrows() {
                let score = scores[[i, 0]];
                if score < config.score_threshold { continue; }

                let dist = bboxes.slice(s![i, ..]);
                let anchor = anchors.slice(s![i, ..]);

                let x1 = (dist[0].mul_add(-(stride as f32), anchor[0]) - params.x_offset) / params.ratio;
                let y1 = (dist[1].mul_add(-(stride as f32), anchor[1]) - params.y_offset) / params.ratio;
                let x2 = (dist[2].mul_add(stride as f32, anchor[0]) - params.x_offset) / params.ratio;
                let y2 = (dist[3].mul_add(stride as f32, anchor[1]) - params.y_offset) / params.ratio;

                let kps_dist = kps.slice(s![i, ..]);
                let mut landmarks = Vec::with_capacity(5);
                for j in 0..5 {
                    landmarks.push((
                        (kps_dist[j * 2].mul_add(stride as f32, anchor[0]) - params.x_offset) / params.ratio,
                        (kps_dist[j * 2 + 1].mul_add(stride as f32, anchor[1]) - params.y_offset) / params.ratio,
                    ));
                }
                candidate_faces.push(Face {
                    bbox: BBox { x1, y1, x2, y2 },
                    landmarks: Some(landmarks),
                    score,
                });
            }
        }
        Ok(Self::apply_nms(candidate_faces, config.iou_threshold))
    }

    fn apply_nms(mut faces: Vec<Face>, iou_threshold: f32) -> Vec<Face> {
        faces.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });        let mut suppressed = vec![false; faces.len()];
        for i in 0..faces.len() {
            if suppressed[i] { continue; }
            for j in (i + 1)..faces.len() {
                if suppressed[j] { continue; }
                if Self::calculate_iou(&faces[i].bbox, &faces[j].bbox) > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
        let mut idx = 0;
        faces.retain(|_| {
            let keep = !suppressed[idx];
            idx += 1;
            keep
        });
        faces
    }

    fn calculate_iou(a: &BBox, b: &BBox) -> f32 {
        let x1 = a.x1.max(b.x1);
        let y1 = a.y1.max(b.y1);
        let x2 = a.x2.min(b.x2);
        let y2 = a.y2.min(b.y2);
        let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        if intersection <= 0.0 { return 0.0; }
        intersection / (a.area() + b.area() - intersection)
    }
}