use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Array2, Array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rust_dl_from_scratch::chapter02::matrix::Matrix;
use rust_dl_from_scratch::chapter02::network::{SimpleNet, SimpleNetMatrix};

fn benchmark_predict_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("Network Predict Small (10x5x3, batch=32)");
    
    // 创建网络
    let net_ndarray = SimpleNet::new(10, 5, 3);
    let net_matrix = SimpleNetMatrix::new(10, 5, 3);
    
    // 生成测试数据
    let input_ndarray = Array::random((32, 10), Uniform::new(-1.0, 1.0));
    let input_matrix = Matrix::from_vec(
        input_ndarray.outer_iter()
            .map(|row| row.to_vec())
            .collect()
    );

    group.bench_function("ndarray", |b| {
        b.iter(|| net_ndarray.predict(black_box(&input_ndarray)))
    });

    group.bench_function("matrix", |b| {
        b.iter(|| net_matrix.predict(black_box(&input_matrix)))
    });

    group.finish();
}

fn benchmark_predict_medium(c: &mut Criterion) {
    let mut group = c.benchmark_group("Network Predict Medium (100x50x10, batch=64)");
    
    let net_ndarray = SimpleNet::new(100, 50, 10);
    let net_matrix = SimpleNetMatrix::new(100, 50, 10);
    
    let input_ndarray = Array::random((64, 100), Uniform::new(-1.0, 1.0));
    let input_matrix = Matrix::from_vec(
        input_ndarray.outer_iter()
            .map(|row| row.to_vec())
            .collect()
    );

    group.bench_function("ndarray", |b| {
        b.iter(|| net_ndarray.predict(black_box(&input_ndarray)))
    });

    group.bench_function("matrix", |b| {
        b.iter(|| net_matrix.predict(black_box(&input_matrix)))
    });

    group.finish();
}

fn benchmark_predict_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("Network Predict Large (784x128x10, batch=128)");
    
    let net_ndarray = SimpleNet::new(784, 128, 10);
    let net_matrix = SimpleNetMatrix::new(784, 128, 10);
    
    let input_ndarray = Array::random((128, 784), Uniform::new(-1.0, 1.0));
    let input_matrix = Matrix::from_vec(
        input_ndarray.outer_iter()
            .map(|row| row.to_vec())
            .collect()
    );

    group.bench_function("ndarray", |b| {
        b.iter(|| net_ndarray.predict(black_box(&input_ndarray)))
    });

    group.bench_function("matrix", |b| {
        b.iter(|| net_matrix.predict(black_box(&input_matrix)))
    });

    group.finish();
}

fn benchmark_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batch Processing (256x784x128x10)");
    
    let net_ndarray = SimpleNet::new(784, 128, 10);
    let net_matrix = SimpleNetMatrix::new(784, 128, 10);
    
    let input_ndarray = Array::random((256, 784), Uniform::new(-1.0, 1.0));
    let input_matrix = Matrix::from_vec(
        input_ndarray.outer_iter()
            .map(|row| row.to_vec())
            .collect()
    );

    group.bench_function("ndarray", |b| {
        b.iter(|| net_ndarray.predict(black_box(&input_ndarray)))
    });

    group.bench_function("matrix", |b| {
        b.iter(|| net_matrix.predict(black_box(&input_matrix)))
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_predict_small,
    benchmark_predict_medium,
    benchmark_predict_large,
    benchmark_batch_processing
);
criterion_main!(benches);