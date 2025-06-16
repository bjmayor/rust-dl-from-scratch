// src/chapter02/activation.rs
use super::matrix::Matrix;

pub fn sigmoid(x: &Matrix) -> Matrix {
    x.map(|v| 1.0 / (1.0 + (-v).exp()))
}

pub fn softmax(x: &Matrix) -> Matrix {
    let mut result = Vec::new();

    for row in &x.data {
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_row: Vec<f64> = row.iter().map(|v| (v - max_val).exp()).collect();
        let sum: f64 = exp_row.iter().sum();
        let softmax_row: Vec<f64> = exp_row.iter().map(|v| v / sum).collect();
        result.push(softmax_row);
    }

    Matrix::from_vec(result)
}
