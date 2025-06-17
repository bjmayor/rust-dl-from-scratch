// src/chapter02/loss.rs
use ndarray::{Array2, Axis};

pub fn mean_squared_error(y: &Array2<f64>, t: &Array2<f64>) -> f64 {
    let diff = y - t;
    let squared_diff = &diff * &diff;
    squared_diff.mean().unwrap()
}

pub fn cross_entropy_error(y: &Array2<f64>, t: &Array2<f64>) -> f64 {
    let delta = 1e-7;
    
    // 防止 log(0)，对 y 加上 delta
    let y_safe = y + delta;
    
    // 计算 -t * log(y)，然后对每个样本求和
    let log_y = y_safe.mapv(|x| x.ln());
    let cross_entropy = -(t * log_y).sum_axis(Axis(1)).mean().unwrap();
    
    cross_entropy
}

// 针对 one-hot 编码优化的交叉熵函数
pub fn cross_entropy_error_optimized(y: &Array2<f64>, t: &Array2<f64>) -> f64 {
    let delta = 1e-7;
    let batch_size = y.nrows() as f64;
    let mut sum = 0.0;
    
    // 只计算真实标签位置的损失
    for (y_row, t_row) in y.outer_iter().zip(t.outer_iter()) {
        for (y_val, t_val) in y_row.iter().zip(t_row.iter()) {
            if *t_val == 1.0 {
                sum += (*y_val + delta).ln();
            }
        }
    }
    
    -sum / batch_size
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mse() {
        let y = array![[0.1, 0.9], [0.8, 0.2]];
        let t = array![[0.0, 1.0], [1.0, 0.0]];
        let loss = mean_squared_error(&y, &t);
        assert!(loss > 0.0 && loss < 1.0);
    }

    #[test]
    fn test_cross_entropy() {
        let y = array![[0.1, 0.9], [0.8, 0.2]];
        let t = array![[0.0, 1.0], [1.0, 0.0]];
        let loss = cross_entropy_error(&y, &t);
        assert!(loss > 0.0 && loss < 3.0);
    }

    #[test]
    fn test_cross_entropy_optimized() {
        let y = array![[0.1, 0.9], [0.8, 0.2]];
        let t = array![[0.0, 1.0], [1.0, 0.0]];
        let loss = cross_entropy_error_optimized(&y, &t);
        assert!(loss > 0.0 && loss < 3.0);
        
        // 两个版本应该给出相近的结果
        let loss_standard = cross_entropy_error(&y, &t);
        assert!((loss - loss_standard).abs() < 1e-10);
    }
}