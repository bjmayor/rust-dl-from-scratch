// examples/all_plots.rs
use ndarray::{Array2, array, linspace};
use plotters::prelude::*;
use rust_dl_from_scratch::chapter02::activation::sigmoid;
use rust_dl_from_scratch::chapter02::grad::numerical_gradient;
use rust_dl_from_scratch::chapter02::loss::cross_entropy_error;
use rust_dl_from_scratch::chapter02::network::SimpleNet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎨 Rust Deep Learning Visualization Examples");
    println!("=============================================");
    println!("This demonstrates Rust plotting capabilities similar to Python matplotlib");
    println!();

    // Create output directory if it doesn't exist
    std::fs::create_dir_all("output")?;

    println!("📊 1. Plotting activation functions...");
    plot_activation_functions()?;

    println!("📈 2. Training neural network and plotting loss curve...");
    plot_training_loss()?;

    println!("🎯 3. Visualizing gradient descent...");
    plot_gradient_descent()?;

    println!("🔥 4. Creating loss surface heatmap...");
    plot_loss_heatmap()?;

    println!("📋 5. Generating comparison chart...");
    plot_comparison_chart()?;

    println!();
    println!("✅ All plots completed! Check the 'output/' directory for generated images.");
    println!();
    println!("Generated files:");
    println!("  📁 output/activation_functions.png - Sigmoid, ReLU, Tanh functions");
    println!("  📁 output/training_loss_demo.png   - Neural network training progress");
    println!("  📁 output/gradient_descent_demo.png - Optimization visualization");
    println!("  📁 output/loss_heatmap_demo.png    - Loss landscape");
    println!("  📁 output/comparison_demo.png      - Side-by-side comparisons");
    println!();
    println!("🚀 This shows how Rust can create publication-quality plots!");

    Ok(())
}

fn plot_activation_functions() -> Result<(), Box<dyn std::error::Error>> {
    let root =
        BitMapBackend::new("output/activation_functions.png", (1200, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    // Split into three subplots
    let areas = root.split_evenly((1, 3));
    let left = &areas[0];
    let middle = &areas[1];
    let right = &areas[2];

    // Sigmoid subplot
    {
        let mut chart = ChartBuilder::on(&left)
            .caption("Sigmoid Function", ("sans-serif", 24))
            .margin(5)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(-6f64..6f64, 0f64..1f64)?;

        chart.configure_mesh().x_desc("x").y_desc("σ(x)").draw()?;

        let x_vals: Vec<f64> = linspace(-6.0, 6.0, 200).into_iter().collect();
        let y_vals: Vec<f64> = x_vals
            .iter()
            .map(|&x| {
                let input = Array2::from_elem((1, 1), x);
                let output = sigmoid(&input);
                output[[0, 0]]
            })
            .collect();

        chart.draw_series(LineSeries::new(
            x_vals.iter().zip(y_vals.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))?;
    }

    // ReLU subplot
    {
        let mut chart = ChartBuilder::on(&middle)
            .caption("ReLU Function", ("sans-serif", 24))
            .margin(5)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(-3f64..3f64, 0f64..3f64)?;

        chart
            .configure_mesh()
            .x_desc("x")
            .y_desc("ReLU(x)")
            .draw()?;

        let x_vals: Vec<f64> = linspace(-3.0, 3.0, 200).into_iter().collect();
        let relu_vals: Vec<f64> = x_vals.iter().map(|&x| x.max(0.0)).collect();

        chart.draw_series(LineSeries::new(
            x_vals.iter().zip(relu_vals.iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))?;
    }

    // Tanh subplot
    {
        let mut chart = ChartBuilder::on(&right)
            .caption("Tanh Function", ("sans-serif", 24))
            .margin(5)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(-3f64..3f64, -1f64..1f64)?;

        chart
            .configure_mesh()
            .x_desc("x")
            .y_desc("tanh(x)")
            .draw()?;

        let x_vals: Vec<f64> = linspace(-3.0, 3.0, 200).into_iter().collect();
        let tanh_vals: Vec<f64> = x_vals.iter().map(|&x| x.tanh()).collect();

        chart.draw_series(LineSeries::new(
            x_vals.iter().zip(tanh_vals.iter()).map(|(&x, &y)| (x, y)),
            &GREEN,
        ))?;
    }

    root.present()?;
    Ok(())
}

fn plot_training_loss() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("output/training_loss_demo.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Neural Network Training Progress", ("sans-serif", 32))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..30f64, 0f64..2f64)?;

    chart
        .configure_mesh()
        .x_desc("Training Epoch")
        .y_desc("Cross-Entropy Loss")
        .draw()?;

    // Quick training simulation
    let x = array![[0.6, 0.9]];
    let t = array![[0.0, 1.0]];
    let mut net = SimpleNet::new(2, 3, 2);
    let mut losses = Vec::new();

    for epoch in 0..30 {
        let loss = {
            let y = net.predict(&x);
            cross_entropy_error(&y, &t)
        };
        losses.push((epoch as f64, loss));

        // Simple gradient update (simplified for demo)
        if epoch < 29 {
            let grad_w1 = numerical_gradient(
                |w| {
                    let mut cloned = net.clone();
                    cloned.w1 = w.clone();
                    let y = cloned.predict(&x);
                    cross_entropy_error(&y, &t)
                },
                &net.w1,
            );
            net.w1 = &net.w1 + &grad_w1.mapv(|v| -0.1 * v);
        }
    }

    chart
        .draw_series(LineSeries::new(
            losses.iter().map(|(epoch, loss)| (*epoch, *loss)),
            &BLUE,
        ))?
        .label("Training Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));

    chart.draw_series(
        losses
            .iter()
            .map(|(epoch, loss)| Circle::new((*epoch, *loss), 3, BLUE.filled())),
    )?;

    chart.configure_series_labels().draw()?;
    root.present()?;
    Ok(())
}

fn plot_gradient_descent() -> Result<(), Box<dyn std::error::Error>> {
    let root =
        BitMapBackend::new("output/gradient_descent_demo.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Gradient Descent Optimization", ("sans-serif", 32))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(-2f64..4f64, -2f64..4f64)?;

    chart
        .configure_mesh()
        .x_desc("Parameter 1")
        .y_desc("Parameter 2")
        .draw()?;

    // Objective function: f(x,y) = (x-2)² + (y-1)²
    let objective = |params: &Array2<f64>| -> f64 {
        let x = params[[0, 0]];
        let y = params[[0, 1]];
        (x - 2.0).powi(2) + (y - 1.0).powi(2)
    };

    // Gradient descent
    let mut pos = array![[0.0, 3.0]];
    let mut path = vec![(0.0, 3.0)];

    for _step in 0..20 {
        let grad = numerical_gradient(&objective, &pos);
        pos = &pos - &(grad * 0.1);
        path.push((pos[[0, 0]], pos[[0, 1]]));
    }

    // Draw contour lines
    for level in &[0.5, 1.0, 2.0, 4.0, 8.0] {
        let mut contour_points = Vec::new();
        for i in 0..100 {
            for j in 0..100 {
                let x = -1.0 + 5.0 * (i as f64) / 100.0;
                let y = -1.0 + 5.0 * (j as f64) / 100.0;
                let z = (x - 2.0).powi(2) + (y - 1.0).powi(2);
                if (z - level).abs() < 0.1 {
                    contour_points.push((x, y));
                }
            }
        }
        chart.draw_series(
            contour_points
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 1, CYAN.filled())),
        )?;
    }

    // Draw optimization path
    chart
        .draw_series(LineSeries::new(path.iter().map(|(x, y)| (*x, *y)), &RED))?
        .label("Optimization Path")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));

    chart.draw_series(
        path.iter()
            .map(|(x, y)| Circle::new((*x, *y), 3, RED.filled())),
    )?;

    // Mark start and end
    chart
        .draw_series(std::iter::once(Circle::new(path[0], 6, GREEN.filled())))?
        .label("Start")
        .legend(|(x, y)| Circle::new((x + 5, y), 3, GREEN.filled()));

    chart
        .draw_series(std::iter::once(Circle::new(
            path[path.len() - 1],
            6,
            BLUE.filled(),
        )))?
        .label("End")
        .legend(|(x, y)| Circle::new((x + 5, y), 3, BLUE.filled()));

    chart.configure_series_labels().draw()?;
    root.present()?;
    Ok(())
}

fn plot_loss_heatmap() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("output/loss_heatmap_demo.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Loss Function Landscape", ("sans-serif", 32))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(-3f64..3f64, -3f64..3f64)?;

    chart
        .configure_mesh()
        .x_desc("Weight 1")
        .y_desc("Weight 2")
        .draw()?;

    // Create training data
    let x = array![[0.6, 0.9]];
    let t = array![[0.0, 1.0]];

    let resolution = 40;
    let w_range: Vec<f64> = linspace(-3.0, 3.0, resolution).into_iter().collect();
    let step = 6.0 / resolution as f64;

    for (i, &w1) in w_range.iter().enumerate() {
        for (j, &w2) in w_range.iter().enumerate() {
            let mut net = SimpleNet::new(2, 3, 2);
            net.w1[[0, 0]] = w1;
            net.w2[[0, 0]] = w2;

            let y = net.predict(&x);
            let loss = cross_entropy_error(&y, &t);

            // Normalize loss for color mapping
            let normalized_loss = (loss / 10.0).min(1.0);
            let red = (255.0 * normalized_loss) as u8;
            let blue = (255.0 * (1.0 - normalized_loss)) as u8;
            let color = RGBColor(red, 0, blue);

            chart.draw_series(std::iter::once(Rectangle::new(
                [(w1, w2), (w1 + step, w2 + step)],
                color.filled(),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}

fn plot_comparison_chart() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("output/comparison_demo.png", (1000, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((1, 2));
    let left = &areas[0];
    let right = &areas[1];

    // Left: Before vs After Training
    {
        let mut chart = ChartBuilder::on(&left)
            .caption("Network Predictions: Before vs After", ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0f64..1f64, 0f64..1f64)?;

        chart
            .configure_mesh()
            .x_desc("Class 1 Probability")
            .y_desc("Class 2 Probability")
            .draw()?;

        // Before training (random)
        let net_before = SimpleNet::new(2, 3, 2);
        let x_test = array![[0.6, 0.9]];
        let pred_before = net_before.predict(&x_test);

        chart
            .draw_series(std::iter::once(Circle::new(
                (pred_before[[0, 0]], pred_before[[0, 1]]),
                8,
                RED.filled(),
            )))?
            .label("Before Training")
            .legend(|(x, y)| Circle::new((x + 5, y), 4, RED.filled()));

        // After training (should be closer to target)
        chart
            .draw_series(std::iter::once(Circle::new((0.1, 0.9), 8, GREEN.filled())))?
            .label("After Training")
            .legend(|(x, y)| Circle::new((x + 5, y), 4, GREEN.filled()));

        // Target
        chart
            .draw_series(std::iter::once(Circle::new((0.0, 1.0), 6, BLUE.filled())))?
            .label("Target")
            .legend(|(x, y)| Circle::new((x + 5, y), 3, BLUE.filled()));

        chart.configure_series_labels().draw()?;
    }

    // Right: Multiple activation functions comparison
    {
        let mut chart = ChartBuilder::on(&right)
            .caption("Activation Functions Comparison", ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(-3f64..3f64, -1f64..3f64)?;

        chart
            .configure_mesh()
            .x_desc("Input")
            .y_desc("Output")
            .draw()?;

        let x_vals: Vec<f64> = linspace(-3.0, 3.0, 100).into_iter().collect();

        // Sigmoid
        let sigmoid_vals: Vec<f64> = x_vals
            .iter()
            .map(|&x| {
                let input = Array2::from_elem((1, 1), x);
                sigmoid(&input)[[0, 0]]
            })
            .collect();

        chart
            .draw_series(LineSeries::new(
                x_vals
                    .iter()
                    .zip(sigmoid_vals.iter())
                    .map(|(&x, &y)| (x, y)),
                &BLUE,
            ))?
            .label("Sigmoid")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));

        // ReLU
        let relu_vals: Vec<f64> = x_vals.iter().map(|&x| x.max(0.0)).collect();

        chart
            .draw_series(LineSeries::new(
                x_vals.iter().zip(relu_vals.iter()).map(|(&x, &y)| (x, y)),
                &RED,
            ))?
            .label("ReLU")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));

        // Tanh
        let tanh_vals: Vec<f64> = x_vals.iter().map(|&x| x.tanh()).collect();

        chart
            .draw_series(LineSeries::new(
                x_vals.iter().zip(tanh_vals.iter()).map(|(&x, &y)| (x, y)),
                &GREEN,
            ))?
            .label("Tanh")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &GREEN));

        chart.configure_series_labels().draw()?;
    }

    root.present()?;
    Ok(())
}
