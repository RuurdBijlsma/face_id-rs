use color_eyre::eyre::Result;
use face_id::analyzer::FaceAnalyzer;
use std::path::Path;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(
            "info,ort=warn,xet_client=warn,xet_data=warn,xet_runtime=warn,xet=warn",
        ))
        .init();

    let analyzer = FaceAnalyzer::from_hf()
        .cache_dir(Path::new("MY_CACHE"))
        .build()
        .await?;
    let faces = analyzer.analyze(&image::open("assets/img/obamna.jpg")?)?;

    for (i, face) in faces.iter().enumerate() {
        println!("Face {i}");
        println!("    Box: {:?}", face.detection.bbox);
        println!("    Score: {:?}", face.detection.score); // Confidence score of detection
        println!("    Landmarks: {:?}", face.detection.landmarks); // location of eyes, mouth, nose

        println!("    Gender: {:?}", face.gender);
        println!("    Age: {:?}", face.age);
        println!("    Embedding [..5]: {:?}", &face.embedding[..5]);
    }

    Ok(())
}
