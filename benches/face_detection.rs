use criterion::{criterion_group, criterion_main, Criterion};
use detect_faces_test::detector::{DetectorConfig, ScrfdDetector};
use ort::value::Value;
use std::hint::black_box;

const MODEL_PATH: &str = "assets/models/34g_gnkps.onnx";
const IMG_PATH: &str = "assets/img/hink.jpg";

/// Benchmarks the initialization of the detector (loading model + anchor generation)
fn bench_init(c: &mut Criterion) {
    c.bench_function("0_detector_new", |b| {
        b.iter(|| {
            let _ = ScrfdDetector::new(
                black_box(MODEL_PATH),
                black_box(DetectorConfig::default())
            ).unwrap();
        })
    });
}

/// Benchmarks the full face detection process
fn bench_full_detect(c: &mut Criterion) {
    let mut detector = ScrfdDetector::new(MODEL_PATH, DetectorConfig::default()).unwrap();
    let img = image::open(IMG_PATH).unwrap();

    c.bench_function("full_pipeline_detect", |b| {
        b.iter(|| {
            let _ = detector.detect(black_box(&img)).unwrap();
        })
    });
}

/// Benchmarks individual components of the pipeline
fn bench_pipeline_steps(c: &mut Criterion) {
    let mut detector = ScrfdDetector::new(MODEL_PATH, DetectorConfig::default()).unwrap();
    let img = image::open(IMG_PATH).unwrap();
    let mut group = c.benchmark_group("pipeline_steps");

    // 1. Preprocessing (Resize + Normalization + Tensor conversion)
    group.bench_function("1_preprocessing", |b| {
        b.iter(|| {
            let (padded, _params) = detector.preprocess(black_box(&img));
            let _tensor = detector.create_input_tensor(black_box(&padded)).unwrap();
        })
    });

    // Setup for Inference
    let (padded, params) = detector.preprocess(&img);
    let input_tensor = detector.create_input_tensor(&padded).unwrap();
    let input_value = Value::from_array(input_tensor).unwrap();

    // 2. Inference (The actual neural network pass)
    group.bench_function("2_inference", |b| {
        b.iter(|| {
            let _ = detector.session.run(ort::inputs![
                &*detector.input_name => input_value.clone()
            ]).unwrap();
        })
    });

    // Setup for Post-processing
    let outputs = detector.session.run(ort::inputs![
        &*detector.input_name => input_value.clone()
    ]).unwrap();

    // 3. Post-processing (BBox decoding + NMS)
    group.bench_function("3_postprocessing", |b| {
        b.iter(|| {
            let _ = ScrfdDetector::postprocess(
                black_box(&outputs),
                black_box(&params),
                black_box(&detector.strides),
                black_box(&detector.anchors),
                black_box(&detector.config),
            ).unwrap();
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_init,
    bench_full_detect,
    bench_pipeline_steps
);
criterion_main!(benches);