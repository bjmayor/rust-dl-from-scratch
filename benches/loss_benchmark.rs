use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Array2, Array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rust_dl_from_scratch::chapter02::matrix::Matrix;
use rust_dl_from_scratch::chapter02::loss::cross_entropy_error_optimized;

// ndarray 版本的损失函数
fn mean_squared_error_ndarray(y: &Array2<f64>, t: &Array2<f64>) -> f64 {
    let diff = y - t;
    let squared_diff = &diff * &diff;
    squared_diff.mean().unwrap()
}

fn cross_entropy_error_ndarray(y: &Array2<f64>, t: &Array2<f64>) -> f64 {
    let delta = 1e-7;
    let y_safe = y + delta;
    let log_y = y_safe.mapv(|x| x.ln());
    let cross_entropy = -(t * log_y).sum_axis(ndarray::Axis(1)).mean().unwrap();
    cross_entropy
}

// Matrix 版本的损失函数
fn mean_squared_error_matrix(y: &Matrix, t: &Matrix) -> f64 {
    let mut sum = 0.0;
    for (y_row, t_row) in y.data.iter().zip(t.data.iter()) {
        for (y_val, t_val) in y_row.iter().zip(t_row.iter()) {
            sum += (y_val - t_val).powi(2);
        }
    }
    sum / (y.rows as f64)
}

fn cross_entropy_error_matrix(y: &Matrix, t: &Matrix) -> f64 {
    let delta = 1e-7;
    let batch_size = y.rows;
    let mut sum = 0.0;

    for (y_row, t_row) in y.data.iter().zip(t.data.iter()) {
        for (y_val, t_val) in y_row.iter().zip(t_row.iter()) {
            if *t_val == 1.0 {
                sum += (*y_val + delta).ln();
            }
        }
    }

    -sum / (batch_size as f64)
}

fn benchmark_mse_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("MSE Small (100x10)");
    
    // 生成测试数据
    let y_ndarray = Array::random((100, 10), Uniform::new(0.0, 1.0));
    let t_ndarray = Array::random((100, 10), Uniform::new(0.0, 1.0));
    
    let y_matrix = Matrix::from_vec(
        y_ndarray.outer_iter()
            .map(|row| row.to_vec())
            .collect()
    );
    let t_matrix = Matrix::from_vec(
        t_ndarray.outer_iter()
            .map(|row| row.to_vec())
            .collect()
    );

    group.bench_function("ndarray", |b| {
        b.iter(|| mean_squared_error_ndarray(black_box(&y_ndarray), black_box(&t_ndarray)))
    });

    group.bench_function("matrix", |b| {
        b.iter(|| mean_squared_error_matrix(black_box(&y_matrix), black_box(&t_matrix)))
    });

    group.finish();
}

fn benchmark_mse_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("MSE Large (1000x100)");
    
    let y_ndarray = Array::random((1000, 100), Uniform::new(0.0, 1.0));
    let t_ndarray = Array::random((1000, 100), Uniform::new(0.0, 1.0));
    
    let y_matrix = Matrix::from_vec(
        y_ndarray.outer_iter()
            .map(|row| row.to_vec())
            .collect()
    );
    let t_matrix = Matrix::from_vec(
        t_ndarray.outer_iter()
            .map(|row| row.to_vec())
            .collect()
    );

    group.bench_function("ndarray", |b| {
        b.iter(|| mean_squared_error_ndarray(black_box(&y_ndarray), black_box(&t_ndarray)))
    });

    group.bench_function("matrix", |b| {
        b.iter(|| mean_squared_error_matrix(black_box(&y_matrix), black_box(&t_matrix)))
    });

    group.finish();
}

fn benchmark_cross_entropy_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cross Entropy Small (100x10)");
    
    // 生成one-hot编码的标签
    let mut y_data = Vec::new();
    let mut t_data = Vec::new();
    
    for _ in 0..100 {
        let mut y_row = vec![0.1; 10];
        let mut t_row = vec![0.0; 10];
        let true_class = (rand::random::<f64>() * 10.0) as usize;
        y_row[true_class] = 0.9;
        t_row[true_class] = 1.0;
        y_data.push(y_row);
        t_data.push(t_row);
    }
    
    let y_ndarray = Array2::from_shape_vec((100, 10), y_data.concat()).unwrap();
    let t_ndarray = Array2::from_shape_vec((100, 10), t_data.concat()).unwrap();
    
    let y_matrix = Matrix::from_vec(y_data);
    let t_matrix = Matrix::from_vec(t_data);

    group.bench_function("ndarray", |b| {
        b.iter(|| cross_entropy_error_ndarray(black_box(&y_ndarray), black_box(&t_ndarray)))
    });

    group.bench_function("matrix", |b| {
        b.iter(|| cross_entropy_error_matrix(black_box(&y_matrix), black_box(&t_matrix)))
    });

    group.bench_function("ndarray_optimized", |b| {
        b.iter(|| cross_entropy_error_optimized(black_box(&y_ndarray), black_box(&t_ndarray)))
    });

    group.finish();
}

fn benchmark_cross_entropy_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cross Entropy Large (1000x100)");
    
    let mut y_data = Vec::new();
    let mut t_data = Vec::new();
    
    for _ in 0..1000 {
        let mut y_row = vec![0.01; 100];
        let mut t_row = vec![0.0; 100];
        let true_class = (rand::random::<f64>() * 100.0) as usize;
        y_row[true_class] = 0.99;
        t_row[true_class] = 1.0;
        y_data.push(y_row);
        t_data.push(t_row);
    }
    
    let y_ndarray = Array2::from_shape_vec((1000, 100), y_data.concat()).unwrap();
    let t_ndarray = Array2::from_shape_vec((1000, 100), t_data.concat()).unwrap();
    
    let y_matrix = Matrix::from_vec(y_data);
    let t_matrix = Matrix::from_vec(t_data);

    group.bench_function("ndarray", |b| {
        b.iter(|| cross_entropy_error_ndarray(black_box(&y_ndarray), black_box(&t_ndarray)))
    });

    group.bench_function("matrix", |b| {
        b.iter(|| cross_entropy_error_matrix(black_box(&y_matrix), black_box(&t_matrix)))
    });

    group.bench_function("ndarray_optimized", |b| {
        b.iter(|| cross_entropy_error_optimized(black_box(&y_ndarray), black_box(&t_ndarray)))
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_mse_small,
    benchmark_mse_large,
    benchmark_cross_entropy_small,
    benchmark_cross_entropy_large
);
criterion_main!(benches);