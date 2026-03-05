use ndarray::{s, Array2, Array3, Array4, ArrayD, ArrayViewD, Axis};
use opencv::core::{self, Mat, Scalar, Vector};
use opencv::dnn;
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::MatTraitConst;
use ort::execution_providers::CPUExecutionProvider;
use ort::session::Session;
use ort::value::Value;
use std::collections::HashMap;
use std::cmp::Ordering;
use std::error::Error;
use std::env;
use std::path::Path;

// ============================================================================
// OpenCVHelper
// ============================================================================
struct OpenCVHelper {
    mean: f32,
    std: f32,
}

impl OpenCVHelper {
    fn new(mean: f32, std: f32) -> Self {
        Self { mean, std }
    }

    fn prepare_input_tensor(
        &self,
        image: &Mat,
        input_size: (i32, i32),
    ) -> Result<Array4<f32>, Box<dyn Error>> {
        let blob = dnn::blob_from_image(
            &image,
            1.0 / self.std as f64,
            core::Size::new(input_size.0 as i32, input_size.1 as i32),
            core::Scalar::new(self.mean as f64, self.mean as f64, self.mean as f64, 0.0),
            true,
            false,
            core::CV_32F,
        )?;

        let tensor_shape = (1, 3, input_size.1 as usize, input_size.0 as usize);
        let tensor_data: Vec<f32> = blob.data_typed()?.to_vec();
        let input_tensor = Array4::from_shape_vec(tensor_shape, tensor_data)?;

        Ok(input_tensor)
    }

    fn resize_with_aspect_ratio(
        &self,
        image: &Mat,
        target_size: (i32, i32),
    ) -> Result<(Mat, f32, i32, i32), Box<dyn Error>> {
        let orig_width = image.cols() as f32;
        let orig_height = image.rows() as f32;

        let (input_width, input_height) = target_size;
        let im_ratio = orig_height / orig_width;
        let model_ratio = input_height as f32 / input_width as f32;

        let (new_width, new_height, x_offset, y_offset) = if im_ratio > model_ratio {
            let new_height = input_height;
            let new_width = ((input_height as f32) / im_ratio).round() as i32;
            let x_offset = (input_width - new_width) / 2;
            (new_width, new_height, x_offset, 0)
        } else {
            let new_width = input_width;
            let new_height = ((input_width as f32) * im_ratio).round() as i32;
            let y_offset = (input_height - new_height) / 2;
            (new_width, new_height, 0, y_offset)
        };

        let det_scale = new_height as f32 / orig_height;

        let mut opencv_resized_image = core::Mat::default();
        opencv::imgproc::resize(
            &image,
            &mut opencv_resized_image,
            core::Size::new(new_width, new_height),
            0.0,
            0.0,
            opencv::imgproc::INTER_LINEAR,
        )?;

        let mut det_image = core::Mat::new_rows_cols_with_default(
            input_height,
            input_width,
            core::CV_8UC3,
            core::Scalar::all(0.0),
        )?;
        let mut roi = det_image.roi_mut(core::Rect::new(x_offset, y_offset, new_width, new_height))?;
        opencv_resized_image.copy_to(&mut roi)?;

        Ok((det_image, det_scale, x_offset, y_offset))
    }
}

// ============================================================================
// ScrfdHelpers
// ============================================================================
struct ScrfdHelpers;

impl ScrfdHelpers {
    fn distance2bbox(
        points: &Array2<f32>,
        distance: &Array2<f32>,
        max_shape: Option<(usize, usize)>,
    ) -> Array2<f32> {
        let mut x1 = &points.column(0) - &distance.column(0);
        let mut y1 = &points.column(1) - &distance.column(1);
        let mut x2 = &points.column(0) + &distance.column(2);
        let mut y2 = &points.column(1) + &distance.column(3);

        let (x1, y1, x2, y2) = if let Some((height, width)) = max_shape {
            let width = width as f32;
            let height = height as f32;
            x1.mapv_inplace(|x| x.max(0.0).min(width));
            y1.mapv_inplace(|y| y.max(0.0).min(height));
            x2.mapv_inplace(|x| x.max(0.0).min(width));
            y2.mapv_inplace(|y| y.max(0.0).min(height));
            (x1, y1, x2, y2)
        } else {
            (x1, y1, x2, y2)
        };

        let concatenated =
            ndarray::stack(Axis(1), &[x1.view(), y1.view(), x2.view(), y2.view()]).unwrap();
        concatenated
    }

    fn distance2kps(
        points: &Array2<f32>,
        distance: &Array2<f32>,
        max_shape: Option<(usize, usize)>,
    ) -> Array2<f32> {
        let num_keypoints = distance.shape()[1] / 2;
        let mut preds = Vec::with_capacity(2 * num_keypoints);

        for i in 0..num_keypoints {
            let mut px = &points.column(0) + &distance.column(2 * i);
            let mut py = &points.column(1) + &distance.column(2 * i + 1);
            let (px, py) = if let Some((height, width)) = max_shape {
                let width = width as f32;
                let height = height as f32;
                px.mapv_inplace(|x| x.max(0.0).min(width));
                py.mapv_inplace(|y| y.max(0.0).min(height));
                (px, py)
            } else {
                (px, py)
            };
            preds.push(px.insert_axis(Axis(1)));
            preds.push(py.insert_axis(Axis(1)));
        }

        ndarray::concatenate(Axis(1), &preds.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap()
    }

    fn nms(dets: &Array2<f32>, iou_thres: f32) -> Vec<usize> {
        if dets.is_empty() {
            return Vec::new();
        }
        let x1 = dets.column(0);
        let y1 = dets.column(1);
        let x2 = dets.column(2);
        let y2 = dets.column(3);
        let scores = dets.column(4);

        let areas = (&x2 - &x1 + 1.0) * (&y2 - &y1 + 1.0);
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_unstable_by(|&i, &j| {
            scores[j].partial_cmp(&scores[i]).unwrap_or(Ordering::Equal)
        });

        let mut keep = Vec::new();
        while !order.is_empty() {
            let i = order[0];
            keep.push(i);

            if order.len() == 1 {
                break;
            }

            let order_rest = &order[1..];

            let x1_i = x1[i];
            let y1_i = y1[i];
            let x2_i = x2[i];
            let y2_i = y2[i];
            let area_i = areas[i];

            let mut x1_order = x1.select(Axis(0), order_rest);
            let mut y1_order = y1.select(Axis(0), order_rest);
            let mut x2_order = x2.select(Axis(0), order_rest);
            let mut y2_order = y2.select(Axis(0), order_rest);
            let areas_order = areas.select(Axis(0), order_rest);

            x1_order.mapv_inplace(|x| x1_i.max(x));
            y1_order.mapv_inplace(|y| y1_i.max(y));
            x2_order.mapv_inplace(|x| x2_i.min(x));
            y2_order.mapv_inplace(|y| y2_i.min(y));
            let (xx1, yy1, xx2, yy2) = (x1_order, y1_order, x2_order, y2_order);

            let mut w = &xx2 - &xx1 + 1.0;
            w.mapv_inplace(|x| x.max(0.0));
            let mut h = &yy2 - &yy1 + 1.0;
            h.mapv_inplace(|y| y.max(0.0));
            let inter = &w * &h;
            let ovr = &inter / (area_i + &areas_order - &inter);

            let inds: Vec<usize> = ovr
                .iter()
                .enumerate()
                .filter(|&(_, &ov)| ov <= iou_thres)
                .map(|(idx, _)| idx)
                .collect();

            let mut new_order = Vec::with_capacity(inds.len());
            for &idx in &inds {
                new_order.push(order[idx + 1]);
            }
            order = new_order;
        }
        keep
    }

    fn generate_anchor_centers(
        num_anchors: usize,
        height: usize,
        width: usize,
        stride: f32,
    ) -> Array2<f32> {
        let mut anchor_centers = Array2::zeros((height * width, 2));

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                anchor_centers[[idx, 0]] = x as f32;
                anchor_centers[[idx, 1]] = y as f32;
            }
        }

        anchor_centers.mapv_inplace(|x| x * stride);

        let anchor_centers = if num_anchors > 1 {
            let mut repeated_anchors = Array2::zeros((height * width * num_anchors, 2));

            for (i, row) in anchor_centers.rows().into_iter().enumerate() {
                for j in 0..num_anchors {
                    repeated_anchors
                        .slice_mut(ndarray::s![i * num_anchors + j, ..])
                        .assign(&row);
                }
            }

            repeated_anchors
        } else {
            anchor_centers
        };

        anchor_centers
    }

    fn concatenate_array2(arrays: &[Array2<f32>]) -> Result<Array2<f32>, Box<dyn Error>> {
        if arrays.is_empty() {
            return Ok(Array2::<f32>::zeros((0, 0)));
        }
        Ok(ndarray::concatenate(
            Axis(0),
            &arrays.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )?)
    }

    fn concatenate_array3(arrays: &[Array3<f32>]) -> Result<Array3<f32>, Box<dyn Error>> {
        if arrays.is_empty() {
            return Ok(Array3::<f32>::zeros((0, 0, 0)));
        }
        Ok(ndarray::concatenate(
            Axis(0),
            &arrays.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )?)
    }
}

// ============================================================================
// RelativeConversion
// ============================================================================
struct RelativeConversion;

impl RelativeConversion {
    fn absolute_to_relative_bboxes(
        bboxes: &Array2<f32>,
        img_width: u32,
        img_height: u32,
    ) -> Array2<f32> {
        let img_width_f = img_width as f32;
        let img_height_f = img_height as f32;

        let ncols = bboxes.ncols();
        let mut relative_bboxes = Array2::<f32>::zeros((bboxes.nrows(), ncols));

        for (i, bbox) in bboxes.axis_iter(Axis(0)).enumerate() {
            let x1 = bbox[0];
            let y1 = bbox[1];
            let x2 = bbox[2];
            let y2 = bbox[3];

            relative_bboxes[[i, 0]] = x1 / img_width_f;
            relative_bboxes[[i, 1]] = y1 / img_height_f;
            relative_bboxes[[i, 2]] = (x2 - x1) / img_width_f;
            relative_bboxes[[i, 3]] = (y2 - y1) / img_height_f;

            for c in 4..ncols {
                relative_bboxes[[i, c]] = bbox[c];
            }
        }

        relative_bboxes
    }

    fn absolute_to_relative_keypoints(
        keypoints: &Array3<f32>,
        img_width: u32,
        img_height: u32,
    ) -> Array3<f32> {
        let img_width_f = img_width as f32;
        let img_height_f = img_height as f32;

        let mut relative_keypoints = Array3::<f32>::zeros(keypoints.dim());

        for (i, kp_set) in keypoints.axis_iter(Axis(0)).enumerate() {
            for (j, kp) in kp_set.axis_iter(Axis(0)).enumerate() {
                let x_rel = kp[0] / img_width_f;
                let y_rel = kp[1] / img_height_f;

                relative_keypoints[[i, j, 0]] = x_rel.clamp(0.0, 1.0);
                relative_keypoints[[i, j, 1]] = y_rel.clamp(0.0, 1.0);
            }
        }

        relative_keypoints
    }
}

// ============================================================================
// SCRFD
// ============================================================================
struct SCRFD {
    input_size: (i32, i32),
    conf_thres: f32,
    iou_thres: f32,
    _fmc: usize,
    feat_stride_fpn: Vec<i32>,
    num_anchors: usize,
    use_kps: bool,
    opencv_helper: OpenCVHelper,
    session: Session,
    input_names: Vec<String>,
    relative_output: bool,
}

impl SCRFD {
    fn new(
        session: Session,
        input_size: (i32, i32),
        conf_thres: f32,
        iou_thres: f32,
        relative_output: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let fmc = 3;
        let feat_stride_fpn = vec![8, 16, 32];
        let num_anchors = 2;
        let use_kps = true;

        let mean = 127.5;
        let std = 128.0;

        let input_names = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();

        Ok(SCRFD {
            input_size,
            conf_thres,
            iou_thres,
            _fmc: fmc,
            feat_stride_fpn,
            num_anchors,
            use_kps,
            opencv_helper: OpenCVHelper::new(mean, std),
            session,
            input_names,
            relative_output,
        })
    }

    fn forward(
        &mut self,
        input_tensor: ArrayD<f32>,
        center_cache: &mut HashMap<(i32, i32, i32), Array2<f32>>,
    ) -> Result<(Vec<Array2<f32>>, Vec<Array2<f32>>, Vec<Array3<f32>>), Box<dyn Error>> {
        let mut scores_list = Vec::new();
        let mut bboxes_list = Vec::new();
        let mut kpss_list = Vec::new();
        let input_height = input_tensor.shape()[2];
        let input_width = input_tensor.shape()[3];
        let input_value = Value::from_array(input_tensor)?;
        let input_name = self.input_names[0].clone();
        let input = ort::inputs![input_name => input_value];

        let session_output = match self.session.run(input) {
            Ok(output) => output,
            Err(e) => return Err(Box::new(e)),
        };

        let mut outputs = vec![];
        for (_, output) in session_output.iter().enumerate() {
            let f32_array: ArrayViewD<f32> = output.1.try_extract_array()?;
            outputs.push(f32_array.to_owned());
        }
        drop(session_output);

        let fmc = self._fmc;
        for (idx, &stride) in self.feat_stride_fpn.iter().enumerate() {
            let scores = &outputs[idx];
            let bbox_preds = outputs[idx + fmc].to_shape((outputs[idx + fmc].len() / 4, 4))?;
            let bbox_preds = (bbox_preds * stride as f32).into_owned();
            let kps_preds = (outputs[idx + fmc * 2]
                .to_shape((outputs[idx + fmc * 2].len() / 10, 10))?
                * stride as f32)
                .into_owned();

            let height = input_height / stride as usize;
            let width = input_width / stride as usize;

            let key = (height as i32, width as i32, stride);
            let anchor_centers = center_cache.entry(key).or_insert_with(|| {
                ScrfdHelpers::generate_anchor_centers(
                    self.num_anchors,
                    height,
                    width,
                    stride as f32,
                )
            });

            let pos_inds: Vec<usize> = scores
                .iter()
                .enumerate()
                .filter(|(_, &s)| s > self.conf_thres)
                .map(|(i, _)| i)
                .collect();

            if pos_inds.is_empty() {
                continue;
            }

            let pos_scores = scores.select(Axis(0), &pos_inds);
            let bboxes = ScrfdHelpers::distance2bbox(anchor_centers, &bbox_preds, None);
            let pos_bboxes = bboxes.select(Axis(0), &pos_inds);

            scores_list.push(pos_scores.to_shape((pos_scores.len(), 1))?.to_owned());
            bboxes_list.push(pos_bboxes);

            if self.use_kps {
                let kpss = ScrfdHelpers::distance2kps(anchor_centers, &kps_preds, None);
                let kpss = kpss.to_shape((kpss.shape()[0], kpss.shape()[1] / 2, 2))?;
                let pos_kpss = kpss.select(Axis(0), &pos_inds);
                kpss_list.push(pos_kpss);
            }
        }

        Ok((scores_list, bboxes_list, kpss_list))
    }

    fn detect(
        &mut self,
        image: &Mat,
        max_num: usize,
        metric: &str,
        center_cache: &mut HashMap<(i32, i32, i32), Array2<f32>>,
    ) -> Result<(Array2<f32>, Option<Array3<f32>>), Box<dyn Error>> {
        let orig_width = image.cols() as f32;
        let orig_height = image.rows() as f32;

        let (det_image, det_scale, x_offset, y_offset) = self
            .opencv_helper
            .resize_with_aspect_ratio(image, self.input_size)?;
        let input_tensor = self
            .opencv_helper
            .prepare_input_tensor(&det_image, self.input_size)?;
        let (scores_list, bboxes_list, kpss_list) =
            self.forward(input_tensor.into_dyn(), center_cache)?;

        if scores_list.is_empty() {
            return Err("No faces detected".into());
        }

        let scores = ScrfdHelpers::concatenate_array2(&scores_list)?;
        let mut bboxes = ScrfdHelpers::concatenate_array2(&bboxes_list)?;
        let x_off = x_offset as f32;
        let y_off = y_offset as f32;
        for mut row in bboxes.rows_mut() {
            row[0] = (row[0] - x_off) / det_scale;
            row[1] = (row[1] - y_off) / det_scale;
            row[2] = (row[2] - x_off) / det_scale;
            row[3] = (row[3] - y_off) / det_scale;
        }

        let mut kpss = if self.use_kps {
            let mut kpss = ScrfdHelpers::concatenate_array3(&kpss_list)?;
            for mut face in kpss.outer_iter_mut() {
                for mut kp in face.rows_mut() {
                    kp[0] = (kp[0] - x_off) / det_scale;
                    kp[1] = (kp[1] - y_off) / det_scale;
                }
            }
            Some(kpss)
        } else {
            None
        };

        let scores_ravel = scores.iter().collect::<Vec<_>>();
        let mut order = (0..scores_ravel.len()).collect::<Vec<usize>>();
        order.sort_unstable_by(|&i, &j| {
            scores_ravel[j]
                .partial_cmp(&scores_ravel[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut pre_det = ndarray::concatenate(Axis(1), &[bboxes.view(), scores.view()])?;
        pre_det = pre_det.select(Axis(0), &order);

        let keep = ScrfdHelpers::nms(&pre_det, self.iou_thres);
        let det = pre_det.select(Axis(0), &keep);

        if self.use_kps {
            if let Some(ref mut kpss_array) = kpss {
                *kpss_array = kpss_array.select(Axis(0), &order);
                *kpss_array = kpss_array.select(Axis(0), &keep);
            }
        }

        let det = if max_num > 0 && max_num < det.shape()[0] {
            let area = (&det.slice(s![.., 2]) - &det.slice(s![.., 0]))
                * (&det.slice(s![.., 3]) - &det.slice(s![.., 1]));
            let image_center = (orig_width / 2.0, orig_height / 2.0);
            let offsets = ndarray::stack![
                Axis(0),
                (&det.slice(s![.., 0]) + &det.slice(s![.., 2])) / 2.0 - image_center.1 as f32,
                (&det.slice(s![.., 1]) + &det.slice(s![.., 3])) / 2.0 - image_center.0 as f32,
            ];
            let offset_dist_squared = offsets.mapv(|x| x * x).sum_axis(Axis(0));
            let values = if metric == "max" {
                area.to_owned()
            } else {
                &area - &(offset_dist_squared * 2.0)
            };
            let mut bindex = (0..values.len()).collect::<Vec<usize>>();
            bindex.sort_unstable_by(|&i, &j| {
                values[j]
                    .partial_cmp(&values[i])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            bindex.truncate(max_num);
            let det = det.select(Axis(0), &bindex);
            if self.use_kps {
                if let Some(ref mut kpss_array) = kpss {
                    *kpss_array = kpss_array.select(Axis(0), &bindex);
                }
            }
            det
        } else {
            det
        };

        let bounding_boxes = if self.relative_output {
            RelativeConversion::absolute_to_relative_bboxes(
                &det,
                orig_width as u32,
                orig_height as u32,
            )
        } else {
            det
        };

        let keypoints = if let Some(kpss) = kpss {
            if self.relative_output {
                Some(RelativeConversion::absolute_to_relative_keypoints(
                    &kpss,
                    orig_width as u32,
                    orig_height as u32,
                ))
            } else {
                Some(kpss)
            }
        } else {
            None
        };

        Ok((bounding_boxes, keypoints))
    }
}

// ============================================================================
// Main
// ============================================================================
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "sample_input/1.png".to_string());

    let model_path = "models/det_10g.onnx";

    println!("Loading model from: {}", model_path);
    let session = Session::builder()?
        .with_execution_providers([CPUExecutionProvider::default().build()])?
        .commit_from_file(Path::new(model_path))?;

    println!("Building SCRFD detector...");
    let mut detector = SCRFD::new(session, (640, 640), 0.25, 0.4, true)?;

    println!("Loading image from: {}", image_path);
    let image_data = std::fs::read(&image_path)?;
    let image = match imgcodecs::imdecode(
        &Vector::<u8>::from_slice(&image_data),
        imgcodecs::IMREAD_COLOR,
    ) {
        Ok(img) => img,
        Err(_) => return Err("Failed to decode image".into()),
    };

    println!(
        "Image dimensions: {}x{}",
        image.cols(),
        image.rows()
    );

    println!("\nRunning face detection...");
    let mut center_cache = HashMap::new();
    let (bboxes, keypoints) = detector.detect(&image, 0, "max", &mut center_cache)?;

    println!("\n=== Detection Results ===");
    println!("Number of faces detected: {}", bboxes.nrows());

    if bboxes.nrows() > 0 {
        println!("\nBounding boxes (relative coordinates [left, top, width, height, score]):");
        for (i, bbox) in bboxes.rows().into_iter().enumerate() {
            println!(
                "  Face {}: left={:.4}, top={:.4}, width={:.4}, height={:.4}, score={:.4}",
                i + 1,
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                bbox[4]
            );
        }

        if let Some(kps) = &keypoints {
            println!("\nFacial keypoints (5 points per face, relative coordinates [x, y]):");
            for (i, face_kps) in kps.outer_iter().enumerate() {
                println!("  Face {}:", i + 1);
                for (j, kp) in face_kps.rows().into_iter().enumerate() {
                    println!(
                        "    Keypoint {}: x={:.4}, y={:.4}",
                        j + 1, kp[0], kp[1]
                    );
                }
            }
        }
    } else {
        println!("No faces detected in the image.");
    }

    println!("\n=== Detection Complete ===");

    Ok(())
}
