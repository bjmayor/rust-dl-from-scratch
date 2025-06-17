use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Array2, Array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rust_dl_from_scratch::chapter02::matrix::Matrix;
use rust_dl_from_scratch::chapter02::activation::{sigmoid, softmax, sigmoid_matrix, softmax_matrix};

fn benchmark_sigmoid_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sigmoid Small (100x10)");
    
    // 生成测试数据
    let data_ndarray = Array::random((100, 10), Uniform::new(-5.0, 5.0));
    let data_matrix = Matrix::from_vec(
        data_ndarray.outer_iter()
            .map(|row| row.to_vec())
            .collect()
    );

    group.bench_function("ndarray", |b| {
        b.iter(|| sigmoid(black_box(&data_ndarray)))
    });

    group.bench_function("matrix", |b| {
        b.iter(|| sigmoid_matrix(black_box(&data_matrix)))
    });

    group.finish();
}

fn benchmark_sigmoid_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sigmoid Large (1000x100)");
    
    let data_ndarray = Array::random((1000, 100), Uniform::new(-5.0, 5.0));
    let data_matrix = Matrix::from_vec(
        data_ndarray.outer_iter()
            .map(|row| row.to_vec())
            .collect()
    );

    group.bench_function("ndarray", |b| {
        b.iter(|| sigmoid(black_box(&data_ndarray)))
    });

    group.bench_function("matrix", |b| {
        b.iter(|| sigmoid_matrix(black_box(&data_matrix)))
    });

    group.finish();
}

fn benchmark_softmax_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("Softmax Small (100x10)");
    
    let data_ndarray = Array::random((100, 10), Uniform::new(-5.0, 5.0));
    let data_matrix = Matrix::from_vec(
        data_ndarray.outer_iter()
            .map(|row| row.to_vec())
            .collect()
    );

    group.bench_function("ndarray", |b| {
        b.iter(|| softmax(black_box(&data_ndarray)))
    });

    group.bench_function("matrix", |b| {
        b.iter(|| softmax_matrix(black_box(&data_matrix)))
    });

    group.finish();
}

fn benchmark_softmax_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("Softmax Large (1000x100)");
    
    let data_ndarray = Array::random((1000, 100), Uniform::new(-5.0, 5.0));
    let data_matrix = Matrix::from_vec(
        data_ndarray.outer_iter()
            .map(|row| row.to_vec())
            .collect()
    );

    group.bench_function("ndarray", |b| {
        b.iter(|| softmax(black_box(&data_ndarray)))
    });

    group.bench_function("matrix", |b| {
        b.iter(|| softmax_matrix(black_box(&data_matrix)))
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_sigmoid_small,
    benchmark_sigmoid_large,
    benchmark_softmax_small,
    benchmark_softmax_large
);
criterion_main!(benches);