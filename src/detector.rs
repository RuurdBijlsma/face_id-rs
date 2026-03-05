use crate::error::DetectorError;
use ndarray::{Array2, Array4, ArrayD, Ix2, s};
use opencv::core::{Mat, MatTraitConst, MatTraitConstManual};
use opencv::{core, dnn, imgproc};
use ort::{session::Session, value::Value};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DetectorConfig {
    pub input_size: (i32, i32),
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct PreprocessParams {
    ratio: f32,
    x_offset: i32,
    y_offset: i32,
}

pub struct ScrfdDetector {
    session: Session,
    config: DetectorConfig,
    anchors: Vec<Array2<f32>>,
    strides: Vec<i32>,
}

impl ScrfdDetector {
    pub fn new(
        model_path: impl AsRef<Path>,
        config: DetectorConfig,
    ) -> Result<Self, DetectorError> {
        let session = Session::builder()?.commit_from_file(model_path)?;

        let strides = vec![8, 16, 32];
        let anchors = strides
            .iter()
            .map(|&s| Self::generate_anchors(config.input_size, s))
            .collect();

        Ok(Self {
            session,
            config,
            anchors,
            strides,
        })
    }

    pub fn detect(&mut self, img: &Mat) -> Result<Vec<Face>, DetectorError> {
        let (processed_img, params) = self.preprocess(img)?;
        let input_tensor = self.create_input_tensor(&processed_img)?;
        let input_value = Value::from_array(input_tensor)?;
        let inputs = ort::inputs!["input.1" => input_value];
        let output_tensors: Vec<ArrayD<f32>> = {
            let outputs = self.session.run(inputs)?;

            outputs
                .iter()
                .map(|(_, v)| v.try_extract_array().map(|a| a.to_owned()))
                .collect::<Result<Vec<_>, ort::Error>>()?
        };

        self.postprocess(&output_tensors, &params)
    }

    fn generate_anchors(input_size: (i32, i32), stride: i32) -> Array2<f32> {
        let h = (input_size.1 / stride) as usize;
        let w = (input_size.0 / stride) as usize;
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

    fn preprocess(&self, img: &Mat) -> Result<(Mat, PreprocessParams), DetectorError> {
        let (w_in, h_in) = self.config.input_size;
        let (w_orig, h_orig) = (img.cols(), img.rows());

        let ratio = (w_in as f32 / w_orig as f32).min(h_in as f32 / h_orig as f32);
        let (w_new, h_new) = (
            (w_orig as f32 * ratio) as i32,
            (h_orig as f32 * ratio) as i32,
        );

        let mut resized = Mat::default();
        imgproc::resize(
            img,
            &mut resized,
            core::Size::new(w_new, h_new),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let mut padded =
            Mat::new_rows_cols_with_default(h_in, w_in, img.typ(), core::Scalar::all(0.0))?;
        let x_offset = (w_in - w_new) / 2;
        let y_offset = (h_in - h_new) / 2;

        let mut roi = Mat::roi_mut(
            &mut padded,
            core::Rect::new(x_offset, y_offset, w_new, h_new),
        )?;
        resized.copy_to(&mut roi)?;

        Ok((
            padded,
            PreprocessParams {
                ratio,
                x_offset,
                y_offset,
            },
        ))
    }

    fn create_input_tensor(&self, img: &Mat) -> Result<Array4<f32>, DetectorError> {
        let blob = dnn::blob_from_image(
            img,
            1.0 / 128.0,
            core::Size::new(self.config.input_size.0, self.config.input_size.1),
            core::Scalar::new(127.5, 127.5, 127.5, 0.0),
            true,
            false,
            core::CV_32F,
        )?;

        let data: Vec<f32> = blob.data_typed()?.to_vec();
        let shape = (
            1,
            3,
            self.config.input_size.1 as usize,
            self.config.input_size.0 as usize,
        );
        Ok(Array4::from_shape_vec(shape, data)?)
    }

    fn postprocess(
        &self,
        outputs: &[ArrayD<f32>],
        params: &PreprocessParams,
    ) -> Result<Vec<Face>, DetectorError> {
        let mut candidate_faces = Vec::new();
        let fmc = 3;

        for (idx, &stride) in self.strides.iter().enumerate() {
            let scores = outputs[idx].view().into_dimensionality::<Ix2>()?;
            let bboxes = outputs[idx + fmc].view().into_dimensionality::<Ix2>()?;
            let kps = outputs[idx + fmc * 2].view().into_dimensionality::<Ix2>()?;

            let anchors = &self.anchors[idx];

            for i in 0..scores.nrows() {
                let score = scores[[i, 0]];
                if score < self.config.score_threshold {
                    continue;
                }

                let dist = bboxes.slice(s![i, ..]);
                let anchor = anchors.slice(s![i, ..]);

                let x1 = (dist[0].mul_add(-(stride as f32), anchor[0]) - params.x_offset as f32)
                    / params.ratio;
                let y1 = (dist[1].mul_add(-(stride as f32), anchor[1]) - params.y_offset as f32)
                    / params.ratio;
                let x2 = (dist[2].mul_add(stride as f32, anchor[0]) - params.x_offset as f32)
                    / params.ratio;
                let y2 = (dist[3].mul_add(stride as f32, anchor[1]) - params.y_offset as f32)
                    / params.ratio;

                let kps_dist = kps.slice(s![i, ..]);
                let mut landmarks = Vec::with_capacity(5);
                for j in 0..5 {
                    landmarks.push((
                        (kps_dist[j * 2].mul_add(stride as f32, anchor[0])
                            - params.x_offset as f32)
                            / params.ratio,
                        (kps_dist[j * 2 + 1].mul_add(stride as f32, anchor[1])
                            - params.y_offset as f32)
                            / params.ratio,
                    ));
                }

                candidate_faces.push(Face {
                    bbox: BBox { x1, y1, x2, y2 },
                    landmarks: Some(landmarks),
                    score,
                });
            }
        }

        Ok(self.apply_nms(candidate_faces))
    }

    fn apply_nms(&self, mut faces: Vec<Face>) -> Vec<Face> {
        faces.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        let mut keep = Vec::new();
        let mut active = vec![true; faces.len()];

        for i in 0..faces.len() {
            if !active[i] {
                continue;
            }
            keep.push(faces[i].clone());

            for j in (i + 1)..faces.len() {
                if active[j]
                    && Self::calculate_iou(&faces[i].bbox, &faces[j].bbox)
                        > self.config.iou_threshold
                {
                    active[j] = false;
                }
            }
        }
        keep
    }

    fn calculate_iou(a: &BBox, b: &BBox) -> f32 {
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
