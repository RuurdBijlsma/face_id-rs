use criterion::{Criterion, criterion_group, criterion_main};
use face_id::analyzer::FaceAnalyzer;
use face_id::gender_age::GenderAgeEstimator;
use std::hint::black_box;
use std::time::Duration;
use tokio::runtime::Runtime;

const TEST_IMAGE_FILE: &str = "assets/img/crowd.jpg";

fn block_on<F: Future>(f: F) -> F::Output {
    Runtime::new().unwrap().block_on(f)
}

// fn load_test_images() -> Vec<DynamicImage> {
//     let img_dir = "assets/img";
//     std::fs::read_dir(img_dir)
//         .expect("Image directory not found")
//         .filter_map(|entry| {
//             let path = entry.ok()?.path();
//             let ext = path.extension()?.to_str()?.to_lowercase();
//             if path.is_file() && (ext == "jpg" || ext == "jpeg" || ext == "png") {
//                 Some(image::open(path).expect("Failed to load image"))
//             } else {
//                 None
//             }
//         })
//         .collect()
// }

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("AnalyzeConstructor");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(8));

    let rt = Runtime::new().unwrap();
    group.bench_function("analyzer_construction", |b| {
        b.iter(|| {
            let _ =
                black_box(rt.block_on(async { FaceAnalyzer::from_hf().build().await.unwrap() }));
        });
    });
}

fn bench_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("FullAnalyze");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(17));

    #[cfg(feature = "cuda")]
    let eps = &[face_id::analyzer::default_optimized_cuda()];
    #[cfg(not(feature = "cuda"))]
    let eps = &[];

    let analyzer = block_on(
        FaceAnalyzer::from_hf()
            .with_execution_providers(eps)
            .build(),
    )
    .unwrap();
    let image = &image::open(TEST_IMAGE_FILE).unwrap();

    group.bench_function("analyze_full_pipeline", |b| {
        b.iter(|| {
            let _ = black_box(analyzer.analyze(image).unwrap());
        });
    });
}

fn bench_sub_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("SubComponents");
    group.sample_size(40);
    group.measurement_time(Duration::from_secs(6));

    #[cfg(feature = "cuda")]
    let eps = &[face_id::analyzer::default_optimized_cuda()];
    #[cfg(not(feature = "cuda"))]
    let eps = &[];

    let analyzer = block_on(
        FaceAnalyzer::from_hf()
            .with_execution_providers(eps)
            .build(),
    )
    .unwrap();
    let image = image::open(TEST_IMAGE_FILE).unwrap();
    let rgb_image = image.to_rgb8();

    group.bench_function("component_detection_only", |b| {
        b.iter(|| {
            let mut det = analyzer.detector.lock().expect("Mutex poisoned");
            let _ = black_box(det.detect(&image).unwrap());
        });
    });

    // Setup for Alignment + Embedding/GA
    let detection_results = analyzer.analyze(&image).unwrap();
    if detection_results.is_empty() {
        return;
    }

    let face = &detection_results[0];
    let lms_array: [(f32, f32); 5] = face
        .detection
        .landmarks
        .as_ref()
        .expect("No landmarks found")
        .as_slice()
        .try_into()
        .expect("Expected 5 landmarks");

    // 2. Align + Embedding
    group.bench_function("align_plus_embedding", |b| {
        b.iter(|| {
            let aligned = face_id::face_align::norm_crop(&rgb_image, &lms_array, 112);
            let mut emb = analyzer.embedder.lock().expect("Mutex poisoned");
            let _ = black_box(emb.compute_embedding(&aligned).unwrap());
        });
    });

    // 3. Align + Gender/Age
    group.bench_function("align_plus_gender_age", |b| {
        b.iter(|| {
            let cropped = GenderAgeEstimator::align_crop(&rgb_image, &face.detection.bbox, 96);
            let mut ga = analyzer.gender_age.lock().expect("Mutex poisoned");
            let _ = black_box(ga.estimate_batch(&[cropped]).unwrap());
        });
    });
}

// fn bench_parallel_processing(c: &mut Criterion) {
//     let mut group = c.benchmark_group("ParallelProcessing");
//     group.sample_size(20);
//     group.measurement_time(Duration::from_secs(30));
//
//     let analyzer = Arc::new(block_on(FaceAnalyzer::from_hf().build()).unwrap());
//     let images = load_test_images();
//
//     group.bench_function("rayon_parallel_folder_analysis", |b| {
//         b.iter(|| {
//             let results: Vec<_> = images
//                 .par_iter()
//                 .map(|img| analyzer.analyze(img).unwrap())
//                 .collect();
//             black_box(results);
//         });
//     });
// }

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(25)
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(2));
    targets = bench_construction, bench_pipeline, bench_sub_components
);
criterion_main!(benches);
