use ndarray::Array1;

pub fn run_chapter01() {
    println!("Chapter 01: Perceptron");

    let x = Array1::from_vec(vec![0.0, 1.0]);
    let w = Array1::from_vec(vec![0.5, -0.6]);
    let b = 0.1;

    let y = perceptron(&x, &w, b);
    println!("Output: {}", y);
}

fn perceptron(x: &Array1<f64>, w: &Array1<f64>, b: f64) -> f64 {
    let sum = x.dot(w) + b;
    step_function(sum)
}

fn step_function(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

pub fn and_gate(x1: f64, x2: f64) -> f64 {
    let w1 = 0.5;
    let w2 = 0.5;
    let bias = -0.7;

    let tmp = x1 * w1 + x2 * w2 + bias;
    step_function(tmp)
}

pub fn nand_gate(x1: f64, x2: f64) -> f64 {
    let w1 = -0.5;
    let w2 = -0.5;
    let bias = 0.7;

    let tmp = x1 * w1 + x2 * w2 + bias;
    step_function(tmp)
}

pub fn or_gate(x1: f64, x2: f64) -> f64 {
    let w1 = 0.5;
    let w2 = 0.5;
    let bias = -0.2;

    let tmp = x1 * w1 + x2 * w2 + bias;
    step_function(tmp)
}

// 异或门 需要组合多个门来实现
pub fn xor_gate(x1: f64, x2: f64) -> f64 {
    let s1 = nand_gate(x1, x2);
    let s2 = or_gate(x1, x2);
    let y = and_gate(s1, s2);
    y
}

// add test
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_and_gate() {
        let cases = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
        ];

        for (x1, x2, expected) in cases {
            let result = and_gate(x1, x2);
            println!("AND({}, {}) = {}, expected {}", x1, x2, result, expected);
            assert!((result - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_nand_gate() {
        let cases = vec![
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 0.0),
        ];

        for (x1, x2, expected) in cases {
            let result = nand_gate(x1, x2);
            println!("NAND({}, {}) = {}, expected {}", x1, x2, result, expected);
            assert!((result - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_or_gate() {
        let cases = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ];

        for (x1, x2, expected) in cases {
            let result = or_gate(x1, x2);
            println!("OR({}, {}) = {}, expected {}", x1, x2, result, expected);
            assert!((result - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_xor_gate() {
        let cases = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 0.0),
        ];

        for (x1, x2, expected) in cases {
            let result = xor_gate(x1, x2);
            println!("XOR({}, {}) = {}, expected {}", x1, x2, result, expected);
            assert!((result - expected).abs() < 1e-6);
        }
    }
}
