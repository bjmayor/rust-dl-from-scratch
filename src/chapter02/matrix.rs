// src/chapter02/matrix.rs
#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<Vec<f64>>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, val: f64) -> Self {
        Self {
            data: vec![vec![val; cols]; rows],
            rows,
            cols,
        }
    }

    pub fn from_vec(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = data[0].len();
        Self { data, rows, cols }
    }

    pub fn dot(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);
        let mut result = Matrix::new(self.rows, other.cols, 0.0);
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    result.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        result
    }

    /**
     * 两个矩阵相加，支持普通加法和行广播。
     * - 如果形状完全一致，则逐元素相加。
     * - 如果 other 只有一行且列数一致，则对 self 的每一行加上 other 的这一行（行广播）。
     * - 其他情况报错。
     */
    pub fn add(&self, other: &Matrix) -> Matrix {
        if self.rows == other.rows && self.cols == other.cols {
            // 普通逐元素相加
            let mut result = self.clone();
            for i in 0..self.rows {
                for j in 0..self.cols {
                    result.data[i][j] += other.data[i][j];
                }
            }
            result
        } else if other.rows == 1 && self.cols == other.cols {
            // 行广播
            let mut result = self.clone();
            for i in 0..self.rows {
                for j in 0..self.cols {
                    result.data[i][j] += other.data[0][j];
                }
            }
            result
        } else {
            panic!("Matrix add: shape mismatch and not broadcastable");
        }
    }

    pub fn map<F>(&self, func: F) -> Matrix
    where
        F: Fn(f64) -> f64,
    {
        Matrix::from_vec(
            self.data
                .iter()
                .map(|row| row.iter().map(|&x| func(x)).collect())
                .collect(),
        )
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}
