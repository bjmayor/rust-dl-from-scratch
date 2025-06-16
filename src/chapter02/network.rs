// src/chapter02/network.rs
use super::activation::{sigmoid, softmax};
use super::matrix::Matrix;
use rand::rng;
use rand_distr::{Distribution, Normal};

pub struct SimpleNet {
    pub w1: Matrix,
    pub b1: Matrix,
    pub w2: Matrix,
    pub b2: Matrix,
}

impl SimpleNet {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

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
        let z1 = sigmoid(&a1);
        let a2 = z1.dot(&self.w2).add(&self.b2);
        softmax(&a2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot() {
        let a = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::from_vec(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let c = a.dot(&b);
        assert_eq!(c.data, vec![vec![19.0, 22.0], vec![43.0, 50.0],]);
    }

    #[test]
    fn test_sigmoid() {
        let x = Matrix::from_vec(vec![vec![0.0], vec![1.0]]);
        let y = sigmoid(&x);
        assert!((y.data[0][0] - 0.5).abs() < 1e-6);
        assert!((y.data[1][0] - 0.73105).abs() < 1e-4);
    }

    #[test]
    fn test_softmax() {
        let x = Matrix::from_vec(vec![vec![2.0, 1.0, 0.1], vec![1.0, 2.0, 3.0]]);
        let y = softmax(&x);
        for row in y.data {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_predict_shape() {
        let net = SimpleNet::new(3, 5, 2);
        let x = Matrix::from_vec(vec![vec![1.0, 0.5, -1.2], vec![0.0, 0.1, 0.2]]);
        let y = net.predict(&x);
        assert_eq!(y.shape(), (2, 2)); // 2 samples, 2 outputs each
    }

    #[test]
    fn test_predict_sum_1() {
        let net = SimpleNet::new(4, 4, 3);
        let x = Matrix::from_vec(vec![vec![1.0, 2.0, 3.0, 4.0]]);
        let y = net.predict(&x);
        let sum: f64 = y.data[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
