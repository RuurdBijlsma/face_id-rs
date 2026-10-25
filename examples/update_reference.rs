use color_eyre::eyre::Result;
use face_id::analyzer::FaceAnalyzer;
use std::fs;
use std::path::{Path, PathBuf};

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let img_dir = "assets/img";
    let output_json_path = "assets/reference_analysis.json";

    println!("Initializing FaceAnalyzer (downloading/loading models)...");
    let analyzer = FaceAnalyzer::from_hf().build().await?;

    let mut img_paths: Vec<PathBuf> = fs::read_dir(img_dir)?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            if !path.is_file() {
                return false;
            }
            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
            matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png")
        })
        .collect();

    if img_paths.is_empty() {
        eprintln!("No images found in {img_dir}");
        return Ok(());
    }

    // Sort alphabetically for deterministic output order across platforms
    img_paths.sort_unstable();

    println!("Found {} images in: {img_dir}", img_paths.len());
    let mut all_results = Vec::new();
    let mut total_faces = 0;

    for path in img_paths {
        let filename = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        print!("Analyzing {filename}... ");
        let img = image::open(&path)?;
        let analysis_results = analyzer.analyze(&img)?;
        let face_count = analysis_results.len();
        total_faces += face_count;
        println!("found {face_count} face(s)");

        all_results.push(serde_json::json!({
            "filename": filename,
            "results": analysis_results
        }));
    }

    let file = fs::File::create(Path::new(output_json_path))?;
    serde_json::to_writer_pretty(file, &all_results)?;

    println!(
        "\nReference analysis generated successfully!\n- Images: {}\n- Total faces: {}\n- Saved to: {output_json_path}",
        all_results.len(),
        total_faces
    );

    Ok(())
}
