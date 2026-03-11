use crate::error::FaceIdError;
use crate::model_manager::{HfModel, get_hf_model};
use bon::bon;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use ndarray::{Array2, Array4, Ix2, s};
use ort::ep::ExecutionProviderDispatch;
use ort::{
    session::{Session, SessionOutputs},
    value::Value,
};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DetectedFace {
    pub bbox: BoundingBox,
    pub landmarks: Option<Vec<(f32, f32)>>,
    pub score: f32,
}

impl DetectedFace {
    /// Scales relative coordinates back to absolute pixel coordinates.
    #[must_use]
    pub fn to_absolute(&self, width: u32, height: u32) -> Self {
        let w = width as f32;
        let h = height as f32;
        Self {
            bbox: self.bbox.scale(width, height),
            landmarks: self.landmarks.as_ref().map(|lms| {
                lms.iter()
                    .map(|&(x, y)| (x * w, y * h))
                    .collect()
            }),
            ..*self
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BoundingBox {
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

    /// Scales relative coordinates back to absolute pixel coordinates.
    #[must_use]
    pub fn scale(&self, width: u32, height: u32) -> Self {
        let w = width as f32;
        let h = height as f32;
        Self {
            x1: self.x1 * w,
            y1: self.y1 * h,
            x2: self.x2 * w,
            y2: self.y2 * h,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PreprocessParams {
    resized_width: f32,
    resized_height: f32,
    x_offset: f32,
    y_offset: f32,
}

#[derive(Debug, Clone)]
pub struct OutputMap {
    pub stride: i32,
    pub score_name: String,
    pub bbox_name: String,
    pub kps_name: Option<String>,
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
    pub output_maps: Vec<OutputMap>,
    pub input_name: String,
}

#[bon]
impl ScrfdDetector {
    #[builder(finish_fn = build)]
    pub async fn from_hf(
        #[builder(default = HfModel::default_detector())] model: HfModel,
        #[builder(default = (640, 640))] input_size: (u32, u32),
        #[builder(default = 0.5)] score_threshold: f32,
        #[builder(default = 0.4)] iou_threshold: f32,
        #[builder(default = &[])] with_execution_providers: &[ExecutionProviderDispatch],
    ) -> Result<Self, FaceIdError> {
        let model_path = get_hf_model(model).await?;
        Self::builder(model_path)
            .input_size(input_size)
            .score_threshold(score_threshold)
            .iou_threshold(iou_threshold)
            .with_execution_providers(with_execution_providers)
            .build()
    }

    #[builder]
    pub fn new(
        #[builder(start_fn)] model_path: impl AsRef<Path>,
        #[builder(default = (640, 640))] input_size: (u32, u32),
        #[builder(default = 0.5)] score_threshold: f32,
        #[builder(default = 0.4)] iou_threshold: f32,
        #[builder(default = &[])] with_execution_providers: &[ExecutionProviderDispatch],
    ) -> Result<Self, FaceIdError> {
        let session = Session::builder()?
            .with_execution_providers(with_execution_providers)?
            .commit_from_file(model_path)?;
        let input_name = session.inputs()[0].name().to_string();
        let config = DetectorConfig {
            input_size,
            score_threshold,
            iou_threshold,
        };

        let output_maps = Self::parse_output_maps(&session)?;

        let first_map = &output_maps[0];
        let score_output = session
            .outputs()
            .iter()
            .find(|o| o.name() == first_map.score_name)
            .ok_or_else(|| {
                FaceIdError::InvalidModel(format!("Missing output: {}", first_map.score_name))
            })?;

        let num_anchors = score_output.dtype().tensor_shape().map_or(2, |shape| {
            let h = i64::from(config.input_size.1 / first_map.stride as u32);
            let w = i64::from(config.input_size.0 / first_map.stride as u32);
            let total_anchors = if shape.len() > 1 {
                shape.iter().rev().nth(1).copied().unwrap_or(0)
            } else {
                shape.iter().next().copied().unwrap_or(0)
            };
            if h * w == 0 {
                2
            } else if total_anchors > 0 && total_anchors % (h * w) == 0 {
                (total_anchors / (h * w)) as usize
            } else {
                2
            }
        });

        let anchors = output_maps
            .iter()
            .map(|m| Self::generate_anchors(config.input_size, m.stride, num_anchors))
            .collect();

        Ok(Self {
            session,
            config,
            anchors,
            output_maps,
            input_name,
        })
    }

    /// Helper to identify output nodes based on naming conventions or shapes.
    fn parse_output_maps(session: &Session) -> Result<Vec<OutputMap>, FaceIdError> {
        let mut output_maps = Vec::new();
        let has_named = session
            .outputs()
            .iter()
            .any(|o| o.name().starts_with("score_"));

        if has_named {
            let mut strides: Vec<i32> = session
                .outputs()
                .iter()
                .filter_map(|output| output.name().strip_prefix("score_")?.parse::<i32>().ok())
                .collect();
            strides.sort_unstable();

            for stride in strides {
                let kps_name = format!("kps_{stride}");
                let has_kps = session.outputs().iter().any(|o| o.name() == kps_name);
                output_maps.push(OutputMap {
                    stride,
                    score_name: format!("score_{stride}"),
                    bbox_name: format!("bbox_{stride}"),
                    kps_name: if has_kps { Some(kps_name) } else { None },
                });
            }
        } else {
            let mut groups: std::collections::HashMap<i64, (String, String, String)> =
                std::collections::HashMap::new();
            for out in session.outputs() {
                if let Some(shape) = out.dtype().tensor_shape() {
                    let n = if shape.len() > 1 {
                        shape[shape.len() - 2]
                    } else {
                        continue;
                    };
                    let last = shape[shape.len() - 1];
                    let entry = groups
                        .entry(n)
                        .or_insert_with(|| (String::new(), String::new(), String::new()));
                    if last == 1 || last == 2 {
                        entry.0 = out.name().to_string();
                    } else if last == 4 {
                        entry.1 = out.name().to_string();
                    } else if last == 10 || last == 15 {
                        entry.2 = out.name().to_string();
                    }
                }
            }

            let mut n_keys: Vec<i64> = groups.keys().copied().filter(|&k| k > 0).collect();
            n_keys.sort_unstable_by(|a, b| b.cmp(a));

            let mut current_stride = 8;
            for n in n_keys {
                let entry = &groups[&n];
                if !entry.0.is_empty() && !entry.1.is_empty() {
                    output_maps.push(OutputMap {
                        stride: current_stride,
                        score_name: entry.0.clone(),
                        bbox_name: entry.1.clone(),
                        kps_name: if entry.2.is_empty() {
                            None
                        } else {
                            Some(entry.2.clone())
                        },
                    });
                    current_stride *= 2;
                }
            }
        }

        if output_maps.is_empty() {
            return Err(FaceIdError::InvalidModel("No stride info found".into()));
        }

        Ok(output_maps)
    }

    pub fn detect(&mut self, img: &DynamicImage) -> Result<Vec<DetectedFace>, FaceIdError> {
        let (processed_img, params) = self.preprocess(img);
        let input_tensor = self.create_input_tensor(&processed_img)?;
        let input_value = Value::from_array(input_tensor)?;
        let inputs = ort::inputs![&self.input_name => input_value];
        let outputs = self.session.run(inputs)?;

        Self::postprocess(
            &outputs,
            &params,
            &self.output_maps,
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
                resized_width: w_new as f32,
                resized_height: h_new as f32,
                x_offset,
                y_offset,
            },
        )
    }

    pub fn create_input_tensor(
        &self,
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<Array4<f32>, FaceIdError> {
        let (width, height) = img.dimensions();
        let w = width as usize;
        let h = height as usize;
        let raw = img.as_raw();

        let mut array = Array4::<f32>::zeros((1, 3, h, w));

        // Get the entire slice as mutable
        let data = array.as_slice_memory_order_mut().ok_or_else(|| {
            FaceIdError::FailedToGetMutableSlice("Failed to get mutable slice from array".into())
        })?;

        // Buffalo-L SCRFD expects: (x - 127.5) / 128.0
        let channel_stride = h * w;
        for (i, chunk) in raw.chunks_exact(3).enumerate() {
            data[i] = (f32::from(chunk[0]) - 127.5) / 128.0;
            data[i + channel_stride] = (f32::from(chunk[1]) - 127.5) / 128.0;
            data[i + 2 * channel_stride] = (f32::from(chunk[2]) - 127.5) / 128.0;
        }

        Ok(array)
    }

    pub fn postprocess(
        outputs: &SessionOutputs,
        params: &PreprocessParams,
        output_maps: &[OutputMap],
        anchors_list: &[Array2<f32>],
        config: &DetectorConfig,
    ) -> Result<Vec<DetectedFace>, FaceIdError> {
        let mut candidate_faces = Vec::new();

        for (idx, map) in output_maps.iter().enumerate() {
            let scores = Self::extract_and_reshape(outputs, &map.score_name)?;
            let bboxes = Self::extract_and_reshape(outputs, &map.bbox_name)?;
            let kps = if let Some(ref kps_name) = map.kps_name {
                Some(Self::extract_and_reshape(outputs, kps_name)?)
            } else {
                None
            };

            let anchors = &anchors_list[idx];
            let stride_f = map.stride as f32;

            for i in 0..scores.nrows() {
                let score = scores[[i, 0]];
                if score < config.score_threshold {
                    continue;
                }

                let dist = bboxes.slice(s![i, ..]);
                let anchor = anchors.slice(s![i, ..]);
                let anchor_x = anchor[0];
                let anchor_y = anchor[1];

                let x1 = (dist[0].mul_add(-stride_f, anchor_x) - params.x_offset) / params.resized_width;
                let y1 = (dist[1].mul_add(-stride_f, anchor_y) - params.y_offset) / params.resized_height;
                let x2 = (dist[2].mul_add(stride_f, anchor_x) - params.x_offset) / params.resized_width;
                let y2 = (dist[3].mul_add(stride_f, anchor_y) - params.y_offset) / params.resized_height;

                let landmarks = kps.as_ref().map(|kps_tensor| {
                    let kps_dist = kps_tensor.slice(s![i, ..]);
                    let mut lms = Vec::with_capacity(5);
                    for j in 0..5 {
                        let lx = (kps_dist[j * 2].mul_add(stride_f, anchor_x) - params.x_offset)
                            / params.resized_width;
                        let ly = (kps_dist[j * 2 + 1].mul_add(stride_f, anchor_y)
                            - params.y_offset)
                            / params.resized_height;
                        lms.push((lx, ly));
                    }
                    lms
                });

                candidate_faces.push(DetectedFace {
                    bbox: BoundingBox { x1, y1, x2, y2 },
                    landmarks,
                    score,
                });
            }
        }

        Ok(Self::perform_non_maximum_suppression(
            candidate_faces,
            config.iou_threshold,
        ))
    }

    fn perform_non_maximum_suppression(
        mut faces: Vec<DetectedFace>,
        iou_threshold: f32,
    ) -> Vec<DetectedFace> {
        faces.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut kept_faces: Vec<DetectedFace> = Vec::with_capacity(faces.len());
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
    ) -> Result<Array2<f32>, FaceIdError> {
        let array = outputs[key].try_extract_array::<f32>()?;
        if array.ndim() == 3 {
            if array.shape()[0] != 1 {
                return Err(FaceIdError::Ort(format!(
                    "Expected batch size 1 for detector output {key}, got {}",
                    array.shape()[0]
                )));
            }
            Ok(array
                .view()
                .to_shape((array.shape()[1], array.shape()[2]))?
                .to_owned()
                .into_dimensionality::<Ix2>()?)
        } else {
            Ok(array.to_owned().into_dimensionality::<Ix2>()?)
        }
    }

    fn compute_intersection_over_union(a: &BoundingBox, b: &BoundingBox) -> f32 {
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
