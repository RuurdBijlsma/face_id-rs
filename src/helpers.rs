#![allow(clippy::similar_names)]
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use crate::detector::BoundingBox;

/// Extracts a square, padded thumbnail for a face.
///
/// # Arguments
/// * `img` - The source image.
/// * `bbox` - The detected face bounding box.
/// * `padding_factor` - How much context to show. 1.0 = tight crop, 2.0 = face takes up 50% of width. (1.5 - 1.8 is usually ideal for UI).
/// * `size` - The output resolution (e.g., 256 for a 256x256 thumbnail).
#[must_use]
pub fn extract_face_thumbnail(
    img: &DynamicImage,
    bbox: &BoundingBox,
    padding_factor: f32,
    size: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (img_w, img_h) = img.dimensions();

    let width = bbox.width();
    let height = bbox.height();
    let cx = bbox.x1 + width / 2.0;
    let cy = bbox.y1 + height / 2.0;

    // 1. Determine the size of our square crop "side"
    // We use the largest dimension to ensure the face fits if it's tall or wide
    let side = width.max(height) * padding_factor;

    // 2. Calculate the theoretical coordinates of the crop
    let x1 = (cx - side / 2.0).round() as i32;
    let y1 = (cy - side / 2.0).round() as i32;
    let side_u = side.round() as u32;

    // 3. Create a black (or transparent) canvas of the requested 'side' size
    // Using Rgb here to match your crate's patterns, but Rgba could be used for transparency
    let mut canvas = ImageBuffer::new(side_u, side_u);

    // 4. Calculate intersection between the crop and the actual image
    // This handles the "cut off" requirement.
    let src_x1 = x1.max(0) as u32;
    let src_y1 = y1.max(0) as u32;
    let src_x2 = (x1 + side_u.cast_signed()).min(img_w.cast_signed()) as u32;
    let src_y2 = (y1 + side_u.cast_signed()).min(img_h.cast_signed()) as u32;

    if src_x2 > src_x1 && src_y2 > src_y1 {
        let crop_w = src_x2 - src_x1;
        let crop_h = src_y2 - src_y1;

        // Extract the valid part of the image
        let sub_img = img.view(src_x1, src_y1, crop_w, crop_h);

        // 5. Calculate where to paste the image onto the canvas
        // If x1 was negative, the offset will be positive
        let dst_x = (src_x1.cast_signed() - x1) as u64;
        let dst_y = (src_y1.cast_signed() - y1) as u64;

        // Overlay the valid pixels onto the black canvas
        image::imageops::overlay(&mut canvas, &sub_img.to_image(), dst_x.cast_signed(), dst_y.cast_signed());
    }

    // 6. Final resize to the standard thumbnail size
    let dynamic_canvas = DynamicImage::ImageRgba8(canvas);
    dynamic_canvas
        .resize_exact(size, size, image::imageops::FilterType::CatmullRom)
        .to_rgb8()
}