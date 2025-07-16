// src/chapter02/network.rs
use super::activation::{sigmoid, sigmoid_matrix, softmax, softmax_matrix};
use super::matrix::Matrix;
use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

#[derive(Clone)]
pub struct SimpleNet {
    pub w1: Array2<f64>,
    pub b1: Array2<f64>,
    pub w2: Array2<f64>,
    pub b2: Array2<f64>,
}

// 向后兼容的 Matrix 版本
pub struct SimpleNetMatrix {
    pub w1: Matrix,
    pub b1: Matrix,
    pub w2: Matrix,
    pub b2: Matrix,
}

impl SimpleNet {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();

        let w1 = Array::random((input_size, hidden_size), normal);
        let b1 = Array2::zeros((1, hidden_size));
        let w2 = Array::random((hidden_size, output_size), normal);
        let b2 = Array2::zeros((1, output_size));

        Self { w1, b1, w2, b2 }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array2<f64> {
        let a1 = x.dot(&self.w1) + &self.b1;
        let z1 = sigmoid(&a1);
        let a2 = z1.dot(&self.w2) + &self.b2;
        softmax(&a2)
    }
}

impl SimpleNetMatrix {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        use rand::rng;
        use rand_distr::Distribution;

        let mut rng = rng();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

        let w1 = Matrix::from_vec(
            (0..input_size)
                .map(|_| (0..hidden_size).map(|_| normal.sample(&mut rng)).collect())
                .collect(),
        );

        let b1 = Matrix::new(1, hidden_size, 0.0);

        let w2 = Matrix::from_vec(
            (0..hidden_size)
                .map(|_| (0..output_size).map(|_| normal.sample(&mut rng)).collect())
                .collect(),
        );

        let b2 = Matrix::new(1, output_size, 0.0);

        Self { w1, b1, w2, b2 }
    }

    pub fn predict(&self, x: &Matrix) -> Matrix {
        let a1 = x.dot(&self.w1).add(&self.b1);
        let z1 = sigmoid_matrix(&a1);
        let a2 = z1.dot(&self.w2).add(&self.b2);
        softmax_matrix(&a2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dot() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let c = a.dot(&b);
        assert_eq!(c, array![[19.0, 22.0], [43.0, 50.0]]);
    }

    #[test]
    fn test_sigmoid() {
        let x = array![[0.0], [1.0]];
        let y = sigmoid(&x);
        assert!((y[[0, 0]] - 0.5).abs() < 1e-6);
        assert!((y[[1, 0]] - 0.73105).abs() < 1e-4);
    }

    #[test]
    fn test_softmax() {
        let x = array![[2.0, 1.0, 0.1], [1.0, 2.0, 3.0]];
        let y = softmax(&x);
        for row in y.axis_iter(ndarray::Axis(0)) {
            let sum: f64 = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_predict_shape() {
        let net = SimpleNet::new(3, 5, 2);
        let x = array![[1.0, 0.5, -1.2], [0.0, 0.1, 0.2]];
        let y = net.predict(&x);
        assert_eq!(y.shape(), [2, 2]); // 2 samples, 2 outputs each
    }

    #[test]
    fn test_predict_sum_1() {
        let net = SimpleNet::new(4, 4, 3);
        let x = array![[1.0, 2.0, 3.0, 4.0]];
        let y = net.predict(&x);
        let sum: f64 = y.row(0).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    // Matrix 版本的测试
    #[test]
    fn test_matrix_predict_shape() {
        let net = SimpleNetMatrix::new(3, 5, 2);
        let x = Matrix::from_vec(vec![vec![1.0, 0.5, -1.2], vec![0.0, 0.1, 0.2]]);
        let y = net.predict(&x);
        assert_eq!(y.shape(), (2, 2)); // 2 samples, 2 outputs each
    }

    #[test]
    fn test_matrix_predict_sum_1() {
        let net = SimpleNetMatrix::new(4, 4, 3);
        let x = Matrix::from_vec(vec![vec![1.0, 2.0, 3.0, 4.0]]);
        let y = net.predict(&x);
        let sum: f64 = y.data[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
