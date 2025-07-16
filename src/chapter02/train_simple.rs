// src/chapter02/train_simple.rs
use crate::chapter02::grad::numerical_gradient;
use crate::chapter02::loss::cross_entropy_error;
use crate::chapter02::network::SimpleNet;
use ndarray::{Array2, array};

pub fn loss_fn(net: &SimpleNet, x: &Array2<f64>, t: &Array2<f64>) -> f64 {
    let y = net.predict(x);
    cross_entropy_error(&y, t)
}

pub fn train_example() {
    let x = array![[0.6, 0.9]];
    let t = array![[0.0, 1.0]]; // 正确答案是第2类

    let mut net = SimpleNet::new(2, 3, 2); // 2输入 → 3隐藏 → 2输出

    for step in 0..5 {
        let loss_before = loss_fn(&net, &x, &t);
        println!("Step {step} - Loss: {:.6}", loss_before);

        // 计算梯度
        let grad_w1 = numerical_gradient(
            |w| {
                let mut cloned = net.clone();
                cloned.w1 = w.clone();
                loss_fn(&cloned, &x, &t)
            },
            &net.w1,
        );

        let grad_b1 = numerical_gradient(
            |b| {
                let mut cloned = net.clone();
                cloned.b1 = b.clone();
                loss_fn(&cloned, &x, &t)
            },
            &net.b1,
        );

        let grad_w2 = numerical_gradient(
            |w| {
                let mut cloned = net.clone();
                cloned.w2 = w.clone();
                loss_fn(&cloned, &x, &t)
            },
            &net.w2,
        );

        let grad_b2 = numerical_gradient(
            |b| {
                let mut cloned = net.clone();
                cloned.b2 = b.clone();
                loss_fn(&cloned, &x, &t)
            },
            &net.b2,
        );

        // 更新参数
        let lr = 0.1;
        net.w1 = &net.w1 + &grad_w1.mapv(|v| -lr * v);
        net.b1 = &net.b1 + &grad_b1.mapv(|v| -lr * v);
        net.w2 = &net.w2 + &grad_w2.mapv(|v| -lr * v);
        net.b2 = &net.b2 + &grad_b2.mapv(|v| -lr * v);
    }

    let final_loss = loss_fn(&net, &x, &t);
    println!("Final loss: {:.6}", final_loss);
}
