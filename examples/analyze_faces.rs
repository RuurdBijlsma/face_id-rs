use face_id::analyzer::FaceAnalyzer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let analyzer = FaceAnalyzer::from_hf().build().await?;

    let img = image::open("assets/img/obamna.jpg")?;
    let results = analyzer.analyze(&img)?;

    for res in results {
        println!(
            "Face at {:?}, Score: {}",
            res.detection.bbox, res.detection.score
        );
        if let Some(ga) = res.gender_age {
            println!("  Gender: {:?}, Age: {}", ga.gender, ga.age);
        }
        if let Some(emb) = res.embedding {
            println!("  Embedding dims: {}", emb.len());
        }
    }
    Ok(())
}
