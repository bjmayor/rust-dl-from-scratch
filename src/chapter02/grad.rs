// src/chapter02/grad.rs
use ndarray::{Array, Dimension, NdIndex};

#[cfg(test)]
use ndarray::{Ix1, Ix2, arr1, arr2};

const H: f64 = 1e-4;

/// 对一个 f64 标量函数求导
#[allow(dead_code)]
pub fn numerical_diff<F>(f: F, x: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    (f(x + H) - f(x - H)) / (2.0 * H)
}

/// 对一个函数 f(x) 计算其对参数 x 的梯度 (通用维度版本)
pub fn numerical_gradient<F, D>(f: F, x: &Array<f64, D>) -> Array<f64, D>
where
    F: Fn(&Array<f64, D>) -> f64,
    D: Dimension,
    // 我们需要告诉编译器，D 的索引模式 (D::Pattern) 必须是可用于索引维度 D 的类型 (NdIndex<D>)
    // 并且它是可克隆的，因为我们会在循环中多次使用它。
    D::Pattern: NdIndex<D> + Clone,
{
    let mut grad = Array::zeros(x.raw_dim());

    for (i, _val) in x.indexed_iter() {
        let mut xh1 = x.clone();
        let mut xh2 = x.clone();

        // 我们需要克隆 `i`，因为索引操作会消耗（move）它。
        xh1[i.clone()] += H;
        xh2[i.clone()] -= H;

        let fxh1 = f(&xh1);
        let fxh2 = f(&xh2);

        grad[i] = (fxh1 - fxh2) / (2.0 * H);
    }

    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numerical_diff() {
        let f = |x: f64| x.powi(2);
        let dx = numerical_diff(f, 3.0);
        assert!((dx - 6.0).abs() < 1e-3);
    }

    #[test]
    fn test_matrix_gradient() {
        // 测试二维数组
        let f = |x: &Array<f64, Ix2>| x.iter().map(|v| v.powi(2)).sum();
        let x = arr2(&[[3.0, 4.0]]);
        let grad = numerical_gradient(f, &x);
        assert!((grad[[0, 0]] - 6.0).abs() < 1e-3);
        assert!((grad[[0, 1]] - 8.0).abs() < 1e-3);
    }

    #[test]
    fn test_vector_gradient() {
        // 测试一维数组
        let f = |x: &Array<f64, Ix1>| x.iter().map(|v| v.powi(2)).sum();
        let x = arr1(&[3.0, 4.0, 5.0]);
        let grad = numerical_gradient(f, &x);
        assert!((grad[0] - 6.0).abs() < 1e-3);
        assert!((grad[1] - 8.0).abs() < 1e-3);
        assert!((grad[2] - 10.0).abs() < 1e-3);
    }
}
