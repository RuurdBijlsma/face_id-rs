use color_eyre::eyre::Result;
use opencv::core::MatTraitConst;
use opencv::imgcodecs;
use opencv::{core, imgproc};
use detect_faces_test::detector::{DetectorConfig, ScrfdDetector};
use detect_faces_test::error::DetectorError;

fn main() -> Result<()> {
    let image_path = "img/IMG_20200524_183102.jpg";
    let model_path = "models/34g_gnkps.onnx";

    let mut detector = ScrfdDetector::new(model_path, DetectorConfig::default())?;

    let image_bytes = std::fs::read(image_path)?;
    let mut img = imgcodecs::imdecode(
        &core::Vector::from_slice(&image_bytes),
        imgcodecs::IMREAD_COLOR,
    )?;
    if img.empty() {
        return Err(DetectorError::Decode.into());
    }

    let faces = detector.detect(&img)?;
    println!("Detected {} faces", faces.len());

    for face in faces {
        let b = face.bbox;
        imgproc::rectangle(
            &mut img,
            core::Rect::new(
                b.x1 as i32,
                b.y1 as i32,
                b.width() as i32,
                b.height() as i32,
            ),
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;

        if let Some(lms) = face.landmarks {
            for pt in lms {
                imgproc::circle(
                    &mut img,
                    core::Point::new(pt.0 as i32, pt.1 as i32),
                    2,
                    core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                    -1,
                    imgproc::LINE_AA,
                    0,
                )?;
            }
        }
    }

    imgcodecs::imwrite("output.jpg", &img, &core::Vector::new())?;
    Ok(())
}
