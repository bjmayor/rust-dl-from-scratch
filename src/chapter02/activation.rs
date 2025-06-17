// src/chapter02/activation.rs
use ndarray::{Array2, Axis};
use super::matrix::Matrix;

// ndarray 版本的激活函数
pub fn sigmoid(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

pub fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let mut result = x.clone();
    
    // 对每一行进行 softmax 计算
    for mut row in result.axis_iter_mut(Axis(0)) {
        // 数值稳定性：减去最大值
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        row.mapv_inplace(|v| (v - max_val).exp());
        
        // 归一化
        let sum: f64 = row.sum();
        row.mapv_inplace(|v| v / sum);
    }
    
    result
}

// Matrix 版本的激活函数（保持向后兼容）
pub fn sigmoid_matrix(x: &Matrix) -> Matrix {
    x.map(|v| 1.0 / (1.0 + (-v).exp()))
}

pub fn softmax_matrix(x: &Matrix) -> Matrix {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sigmoid() {
        let x = array![[0.0, 1.0], [-1.0, 2.0]];
        let result = sigmoid(&x);
        
        // sigmoid(0) = 0.5
        assert!((result[[0, 0]] - 0.5).abs() < 1e-10);
        // sigmoid(1) ≈ 0.731
        assert!((result[[0, 1]] - 0.7310585786300049).abs() < 1e-10);
        // sigmoid(-1) ≈ 0.269
        assert!((result[[1, 0]] - 0.2689414213699951).abs() < 1e-10);
        // sigmoid(2) ≈ 0.881
        assert!((result[[1, 1]] - 0.8807970779778823).abs() < 1e-10);
    }

    #[test]
    fn test_softmax() {
        let x = array![[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]];
        let result = softmax(&x);
        
        // 每行和应该等于 1
        assert!((result.sum_axis(Axis(1)) - array![1.0, 1.0]).sum().abs() < 1e-10);
        
        // 第二行应该是均匀分布 (所有值相等)
        let row1 = result.row(1);
        assert!((row1[0] - 1.0/3.0).abs() < 1e-10);
        assert!((row1[1] - 1.0/3.0).abs() < 1e-10);
        assert!((row1[2] - 1.0/3.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // 测试大数值的数值稳定性
        let x = array![[1000.0, 1001.0, 1002.0]];
        let result = softmax(&x);
        
        // 应该不会产生 NaN 或 Inf
        assert!(result.iter().all(|&v| v.is_finite()));
        // 和应该等于 1
        assert!((result.sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_matrix() {
        let x = Matrix::from_vec(vec![vec![0.0], vec![1.0]]);
        let y = sigmoid_matrix(&x);
        assert!((y.data[0][0] - 0.5).abs() < 1e-6);
        assert!((y.data[1][0] - 0.73105).abs() < 1e-4);
    }

    #[test]
    fn test_softmax_matrix() {
        let x = Matrix::from_vec(vec![vec![2.0, 1.0, 0.1], vec![1.0, 2.0, 3.0]]);
        let y = softmax_matrix(&x);
        for row in y.data {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}