// examples/plot_training_loss.rs
use ndarray::{Array2, array};
use plotters::prelude::*;
use rust_dl_from_scratch::chapter02::grad::numerical_gradient;
use rust_dl_from_scratch::chapter02::loss::cross_entropy_error;
use rust_dl_from_scratch::chapter02::network::SimpleNet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Training neural network and plotting loss curve...");

    // Create output directory if it doesn't exist
    std::fs::create_dir_all("output")?;

    train_and_plot()?;

    println!("Training loss plot saved to output/training_loss.png");
    Ok(())
}

fn loss_fn(net: &SimpleNet, x: &Array2<f64>, t: &Array2<f64>) -> f64 {
    let y = net.predict(x);
    cross_entropy_error(&y, t)
}

fn train_and_plot() -> Result<(), Box<dyn std::error::Error>> {
    // Training data
    let x = array![[0.6, 0.9]];
    let t = array![[0.0, 1.0]]; // Correct answer is class 2

    let mut net = SimpleNet::new(2, 3, 2); // 2 inputs → 3 hidden → 2 outputs

    let mut losses = Vec::new();
    let epochs = 100;
    let lr = 0.1;

    println!("Training for {} epochs...", epochs);

    for epoch in 0..epochs {
        let loss_before = loss_fn(&net, &x, &t);
        losses.push((epoch as f64, loss_before));

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss_before);
        }

        // Calculate gradients
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

        // Update parameters
        net.w1 = &net.w1 + &grad_w1.mapv(|v| -lr * v);
        net.b1 = &net.b1 + &grad_b1.mapv(|v| -lr * v);
        net.w2 = &net.w2 + &grad_w2.mapv(|v| -lr * v);
        net.b2 = &net.b2 + &grad_b2.mapv(|v| -lr * v);
    }

    let final_loss = loss_fn(&net, &x, &t);
    println!("Final loss: {:.6}", final_loss);

    // Plot the training loss
    plot_loss_curve(&losses)?;

    Ok(())
}

fn plot_loss_curve(losses: &[(f64, f64)]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("output/training_loss.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_loss = losses.iter().map(|(_, loss)| *loss).fold(0.0, f64::max);
    let min_loss = losses
        .iter()
        .map(|(_, loss)| *loss)
        .fold(f64::INFINITY, f64::min);

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Loss Curve", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(
            0f64..(losses.len() as f64),
            (min_loss * 0.9)..(max_loss * 1.1),
        )?;

    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Loss")
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            losses.iter().map(|(epoch, loss)| (*epoch, *loss)),
            &BLUE,
        ))?
        .label("Training Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));

    // Add points for better visibility
    chart.draw_series(
        losses
            .iter()
            .map(|(epoch, loss)| Circle::new((*epoch, *loss), 2, BLUE.filled())),
    )?;

    chart.configure_series_labels().draw()?;
    root.present()?;

    Ok(())
}
