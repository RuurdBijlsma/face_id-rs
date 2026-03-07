#![allow(clippy::similar_names, clippy::many_single_char_names)]

use image::{ImageBuffer, Rgb};
use nalgebra::{ArrayStorage, Matrix2, Matrix2x1, Matrix3, Matrix3x2};

/// Canonical 5-point landmark positions for `ArcFace` at 112×112 resolution.
pub const ARCFACE_DST_112: [(f32, f32); 5] = [
    (38.2946, 51.6963), // left eye
    (73.5318, 51.5014), // right eye
    (56.0252, 71.7366), // nose tip
    (41.5493, 92.3655), // left mouth corner
    (70.7299, 92.2041), // right mouth corner
];

/// Compute the optimal 2D similarity transform (scale + rotation + translation) from `src` to
/// `dst` using the Kabsch–Umeyama algorithm.
///
/// Returns a `Matrix3x2<f32>` where:
/// - Rows 0–1 contain the scaled rotation: `scale * R`
/// - Row 2 contains the translation: `[tx, ty]`
///
/// The equivalent 2×3 affine matrix `M` (as used by `OpenCV`'s `warpAffine`) is:
/// ```text
/// M = | row0[0]  row0[1]  row2[0] |
///     | row1[0]  row1[1]  row2[1] |
/// ```
/// which maps `[x_src, y_src, 1]ᵀ → [x_dst, y_dst]ᵀ`.
pub fn umeyama<const R: usize>(src: &[(f32, f32); R], dst: &[(f32, f32); R]) -> Matrix3x2<f32> {
    let src_x_mean = src.iter().map(|v| v.0).sum::<f32>() / R as f32;
    let src_y_mean = src.iter().map(|v| v.1).sum::<f32>() / R as f32;
    let dst_x_mean = dst.iter().map(|v| v.0).sum::<f32>() / R as f32;
    let dst_y_mean = dst.iter().map(|v| v.1).sum::<f32>() / R as f32;

    // Demean both point sets → results in 2×R matrices (2 rows = [x, y], R cols = points)
    let src_demean_s = ArrayStorage(src.map(|v| [v.0 - src_x_mean, v.1 - src_y_mean]));
    let dst_demean_s = ArrayStorage(dst.map(|v| [v.0 - dst_x_mean, v.1 - dst_y_mean]));
    let src_demean = nalgebra::Matrix::from_array_storage(src_demean_s);
    let dst_demean = nalgebra::Matrix::from_array_storage(dst_demean_s);

    // Cross-covariance matrix A = (dst_demean × src_demeanᵀ) / R  →  2×2
    let a = (dst_demean * src_demean.transpose()) / R as f32;
    let svd = a.svd(true, true);

    let mut d = [1f32; 2];
    if a.determinant() < 0.0 {
        d[1] = -1.0;
    }

    // Rotation matrix t (2×2)
    let mut t = Matrix2::<f32>::identity();
    let s = svd.singular_values;
    let u = svd.u.unwrap_or_else(Matrix2::identity);
    let v = svd.v_t.unwrap_or_else(Matrix2::identity);

    let rank = a.rank(1e-5);

    if rank == 0 {
        // Degenerate case: return identity transform
        return Matrix3x2::identity();
    } else if rank == 1 {
        // Nearly degenerate: only rotate if det is positive
        if u.determinant() * v.determinant() > 0.0 {
            u.mul_to(&v, &mut t);
        } else {
            d[1] = -1.0;
            let dg = Matrix2::new(d[0], 0.0, 0.0, d[1]);
            (u * dg).mul_to(&v, &mut t);
        }
    } else {
        let dg = Matrix2::new(d[0], 0.0, 0.0, d[1]);
        (u * dg).mul_to(&v, &mut t);
    }

    // Scale: dot(d, singular_values) / Var(src)
    // Var(src) = (sum_x² + sum_y²) / R  — population variance of demeaned points
    let d_dot_s = d[0].mul_add(s[0], d[1] * s[1]);
    let var_src = src_demean.remove_row(0).variance() + src_demean.remove_row(1).variance();
    let scale = d_dot_s / var_src;

    // Translation: dst_mean - scale * t * src_mean
    let dst_mean = Matrix2x1::new(dst_x_mean, dst_y_mean);
    let src_mean = Matrix2x1::new(src_x_mean, src_y_mean);
    let translation = dst_mean - scale * t * src_mean;

    let sr = t * scale;
    Matrix3x2::new(
        sr.m11,
        sr.m12,
        sr.m21,
        sr.m22,
        translation[0],
        translation[1],
    )
}

/// Align a face crop to the canonical `ArcFace` pose using the 5 SCRFD keypoints.
///
/// Applies the Umeyama similarity transform that maps the detected keypoints to the
/// canonical [`ARCFACE_DST_112`] positions (scaled to `image_size`), then performs an
/// inverse-affine warp with bilinear interpolation to produce the aligned crop.
///
/// # Parameters
/// - `img` — original full image
/// - `landmarks` — the 5 keypoints from SCRFD: `[left_eye, right_eye, nose, left_mouth, right_mouth]`
/// - `image_size` — output square size in pixels (typically `112` for `ArcFace`, `128` for some variants)
///
/// # Returns
/// A square `image_size × image_size` RGB crop aligned to the canonical face pose.
#[must_use]
pub fn norm_crop(
    img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    landmarks: &[(f32, f32); 5],
    image_size: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let dst = scale_arcface_dst(image_size);
    let m = umeyama(landmarks, &dst);
    warp_affine(img, &m, image_size)
}

/// Scale the canonical `ArcFace` destination points to a given `image_size`.
/// `ArcFace` models supporting sizes divisible by 128 offset x by 8 pixels per the original paper.
fn scale_arcface_dst(image_size: u32) -> [(f32, f32); 5] {
    let ratio;
    let diff_x = if image_size.is_multiple_of(112) {
        ratio = image_size as f32 / 112.0;
        0.0
    } else {
        ratio = image_size as f32 / 128.0;
        8.0 * ratio
    };
    ARCFACE_DST_112.map(|(x, y)| (x.mul_add(ratio, diff_x), y * ratio))
}

/// Apply the affine transform represented by `m` (a `Matrix3x2` from [`umeyama`]) to `img`,
/// producing a square crop of `output_size × output_size` pixels.
///
/// Uses inverse-mapping with bilinear interpolation (same as `OpenCV` `warpAffine`).
fn warp_affine(
    img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    m: &Matrix3x2<f32>,
    output_size: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    // Build the full 3×3 homogeneous matrix from the 3×2 representation.
    // M maps src (original) coords → dst (canonical) coords:
    //   [x_dst, y_dst]ᵀ = M_2x3 * [x_src, y_src, 1]ᵀ
    // where M_2x3 = | row0 | row1 | with row2 = translation
    let mat = Matrix3::new(
        m[(0, 0)],
        m[(0, 1)],
        m[(2, 0)],
        m[(1, 0)],
        m[(1, 1)],
        m[(2, 1)],
        0.0,
        0.0,
        1.0,
    );

    // Invert: M_inv maps canonical coords → original image coords (inverse mapping for the warp)
    let inv = mat.try_inverse().unwrap_or_else(Matrix3::identity);
    let ia = inv[(0, 0)];
    let ib = inv[(0, 1)];
    let itx = inv[(0, 2)];
    let ic = inv[(1, 0)];
    let id = inv[(1, 1)];
    let ity = inv[(1, 2)];

    let (orig_w, orig_h) = img.dimensions();
    let mut output = ImageBuffer::new(output_size, output_size);

    // For every output pixel, find the corresponding source pixel and sample it.
    for py in 0..output_size {
        let py_f = py as f32;
        for px in 0..output_size {
            let px_f = px as f32;
            let sx = ia * px_f + ib * py_f + itx;
            let sy = ic * px_f + id * py_f + ity;
            output.put_pixel(px, py, bilinear_sample(img, sx, sy, orig_w, orig_h));
        }
    }
    output
}

/// Sample a pixel from `img` at floating-point coordinates `(x, y)` using bilinear interpolation.
/// Returns black for out-of-bounds coordinates.
#[inline]
fn bilinear_sample(img: &ImageBuffer<Rgb<u8>, Vec<u8>>, x: f32, y: f32, w: u32, h: u32) -> Rgb<u8> {
    if x < 0.0 || y < 0.0 || x >= w as f32 || y >= h as f32 {
        return Rgb([0, 0, 0]);
    }
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let p00 = img.get_pixel(x0, y0);
    let p10 = img.get_pixel(x1, y0);
    let p01 = img.get_pixel(x0, y1);
    let p11 = img.get_pixel(x1, y1);

    // Bilinear interpolation: lerp along x, then along y
    Rgb([
        bilerp(p00[0], p10[0], p01[0], p11[0], fx, fy),
        bilerp(p00[1], p10[1], p01[1], p11[1], fx, fy),
        bilerp(p00[2], p10[2], p01[2], p11[2], fx, fy),
    ])
}

#[inline]
fn bilerp(c00: u8, c10: u8, c01: u8, c11: u8, fx: f32, fy: f32) -> u8 {
    let top = (f32::from(c10) - f32::from(c00)).mul_add(fx, f32::from(c00));
    let bot = (f32::from(c11) - f32::from(c01)).mul_add(fx, f32::from(c01));
    (bot - top).mul_add(fy, top) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn umeyama_identity() {
        let result = umeyama(&ARCFACE_DST_112, &ARCFACE_DST_112);
        let id = Matrix3x2::identity();
        let res: Vec<f32> = result.iter().copied().collect();
        let exp: Vec<f32> = id.iter().copied().collect();
        for (r, e) in res.iter().zip(exp.iter()) {
            assert!((r - e).abs() < 1e-4, "got {r}, expected {e}");
        }
    }

    /// Shifting all destination points by (+10, +5) should produce a pure translation.
    #[test]
    fn umeyama_translation_only() {
        let src = ARCFACE_DST_112;
        let dst: [(f32, f32); 5] = src.map(|(x, y)| (x + 10.0, y + 5.0));
        let m = umeyama(&src, &dst);
        // Scale ≈ 1, rotation ≈ identity, translation ≈ (10, 5)
        assert!((m[(0, 0)] - 1.0).abs() < 1e-4, "scale_x: {}", m[(0, 0)]);
        assert!((m[(1, 1)] - 1.0).abs() < 1e-4, "scale_y: {}", m[(1, 1)]);
        assert!(m[(0, 1)].abs() < 1e-4, "skew: {}", m[(0, 1)]);
        assert!((m[(2, 0)] - 10.0).abs() < 1e-3, "tx: {}", m[(2, 0)]);
        assert!((m[(2, 1)] - 5.0).abs() < 1e-3, "ty: {}", m[(2, 1)]);
    }

    /// Scaling all destination points by 2× should produce scale ≈ 2, translation ≈ centroid shift.
    #[test]
    fn umeyama_scale_only() {
        let src = ARCFACE_DST_112;
        let dst: [(f32, f32); 5] = src.map(|(x, y)| (x * 2.0, y * 2.0));
        let m = umeyama(&src, &dst);
        // The rotation entries should encode scale ≈ 2
        let scale = m[(0, 0)].hypot(m[(1, 0)]);
        assert!((scale - 2.0).abs() < 1e-3, "scale: {scale}");
    }

    /// The `ARCFACE_DST_112` points aligned to themselves should produce the identity transform.
    #[test]
    fn estimate() {
        let result = umeyama(&ARCFACE_DST_112, &ARCFACE_DST_112);
        assert_eq!(result, Matrix3x2::identity());
    }
}
