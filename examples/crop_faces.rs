use color_eyre::eyre::Result;
use face_id::analyzer::FaceAnalyzer;
use face_id::helpers::extract_face_thumbnail;
use std::fs;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let img_dir = "assets/img";
    let output_dir = Path::new("output_previews/cropped_faces");
    if !output_dir.exists() {
        fs::create_dir_all(output_dir)?;
    }
    let analyzer = FaceAnalyzer::from_hf().build().await?;
    let padding_factor = 1.6;
    let output_size = 256;

    println!("Scanning: {img_dir}");
    for entry in fs::read_dir(img_dir)? {
        let entry = entry?;
        let path = entry.path();
        let img = image::open(&path)?;
        println!("Processing: {}", path.display());
        let faces = analyzer.analyze(&img)?;
        println!("  Found {} faces", faces.len());

        for (i, face) in faces.iter().enumerate() {
            let thumbnail =
                extract_face_thumbnail(&img, &face.detection.bbox, padding_factor, output_size);

            // Save with a name like "crowd_face_0.jpg"
            let filename = path
                .file_name()
                .map_or(String::new(), |f| f.to_string_lossy().to_string());
            let base_folder = output_dir.join(&filename);
            if !base_folder.exists() {
                fs::create_dir_all(&base_folder)?;
            }
            let out_name = format!("face_{i}.jpg");
            let out_path = base_folder.join(out_name);
            thumbnail.save(out_path)?;
        }
    }

    println!("\nDone! Check your crops in: {}", output_dir.display());
    Ok(())
}
