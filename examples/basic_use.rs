use color_eyre::eyre::Result;
use face_id::analyzer::FaceAnalyzer;

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let analyzer = FaceAnalyzer::from_hf().build().await?;
    let faces = analyzer.analyze(&image::open("assets/img/crowd.jpg")?)?;

    for (i, face) in faces.iter().enumerate() {
        println!("Face {i}");
        println!("    Box: {:?}", &face.detection.bbox);
        println!("    Score: {:?}", &face.detection.score); // Confidence score of detection
        println!("    Landmarks: {:?}", &face.detection.landmarks); // location of eyes, mouth, nose

        if let Some(ga) = &face.gender_age {
            println!("    Gender: {:?}", ga.gender);
            println!("    Age: {:?}", ga.age);
        }

        if let Some(x) = &face.embedding {
            println!("    Embedding [..5]: {:?}", &x[..5]);
        }
    }

    Ok(())
}
