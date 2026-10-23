use face_id::analyzer::FaceAnalyzer;

// Its own test binary: ort allows one environment per process, and this test builds it.
#[tokio::test]
async fn analyzer_uses_an_environment_built_by_the_caller() -> color_eyre::Result<()> {
    let environment = ort::init().with_name("caller").build()?;
    let analyzer = FaceAnalyzer::from_hf()
        .with_environment(&environment)
        .build()
        .await?;

    let faces = analyzer.analyze(&image::open("assets/img/da-vinki.jpg")?)?;

    assert_eq!(faces.len(), 1);
    Ok(())
}
