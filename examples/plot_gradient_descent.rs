// examples/plot_gradient_descent.rs
use ndarray::{Array2, linspace};
use plotters::prelude::*;
use rust_dl_from_scratch::chapter02::grad::numerical_gradient;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Visualizing gradient descent on a 2D function...");

    // Create output directory if it doesn't exist
    std::fs::create_dir_all("output")?;

    plot_gradient_descent_2d()?;
    plot_gradient_descent_contour()?;

    println!("Gradient descent plots saved to output/ directory");
    Ok(())
}

// Simple 2D function: f(x, y) = (x-2)² + (y-1)²
fn objective_function(x: f64, y: f64) -> f64 {
    (x - 2.0).powi(2) + (y - 1.0).powi(2)
}

// Wrapper for numerical gradient computation
fn objective_function_array(params: &Array2<f64>) -> f64 {
    objective_function(params[[0, 0]], params[[0, 1]])
}

fn plot_gradient_descent_2d() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("output/gradient_descent_2d.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Gradient Descent Path", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(-1f64..5f64, -2f64..4f64)?;

    chart.configure_mesh().x_desc("x").y_desc("y").draw()?;

    // Starting point
    let mut current_pos = Array2::from_elem((1, 2), 0.0);
    current_pos[[0, 0]] = 0.0; // x
    current_pos[[0, 1]] = 3.0; // y

    let learning_rate = 0.1;
    let num_iterations = 50;
    let mut path = Vec::new();

    // Perform gradient descent
    for i in 0..num_iterations {
        let x = current_pos[[0, 0]];
        let y = current_pos[[0, 1]];
        let loss = objective_function(x, y);

        path.push((x, y));

        if i % 10 == 0 {
            println!(
                "Iteration {}: x={:.3}, y={:.3}, f(x,y)={:.3}",
                i, x, y, loss
            );
        }

        // Calculate gradient
        let grad = numerical_gradient(objective_function_array, &current_pos);

        // Update position
        current_pos = &current_pos - &(grad * learning_rate);
    }

    // Add final point
    path.push((current_pos[[0, 0]], current_pos[[0, 1]]));

    // Draw the path
    chart
        .draw_series(LineSeries::new(path.iter().map(|(x, y)| (*x, *y)), &BLUE))?
        .label("Gradient Descent Path")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));

    // Draw points along the path
    chart.draw_series(path.iter().enumerate().map(|(i, (x, y))| {
        let color = if i == 0 {
            &RED
        } else if i == path.len() - 1 {
            &GREEN
        } else {
            &BLUE
        };
        Circle::new((*x, *y), 3, color.filled())
    }))?;

    // Mark starting and ending points
    chart
        .draw_series(std::iter::once(Circle::new(path[0], 5, RED.filled())))?
        .label("Start")
        .legend(|(x, y)| Circle::new((x + 5, y), 3, RED.filled()));

    chart
        .draw_series(std::iter::once(Circle::new(
            path[path.len() - 1],
            5,
            GREEN.filled(),
        )))?
        .label("End")
        .legend(|(x, y)| Circle::new((x + 5, y), 3, GREEN.filled()));

    // Mark the minimum (2, 1)
    chart
        .draw_series(std::iter::once(Circle::new((2.0, 1.0), 6, BLACK.filled())))?
        .label("True Minimum")
        .legend(|(x, y)| Circle::new((x + 5, y), 3, BLACK.filled()));

    chart.configure_series_labels().draw()?;
    root.present()?;
    println!("Gradient descent path saved to output/gradient_descent_2d.png");
    Ok(())
}

fn plot_gradient_descent_contour() -> Result<(), Box<dyn std::error::Error>> {
    let root =
        BitMapBackend::new("output/gradient_descent_contour.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Gradient Descent with Contour Lines", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(-1f64..5f64, -2f64..4f64)?;

    chart.configure_mesh().x_desc("x").y_desc("y").draw()?;

    // Draw contour lines
    let x_range: Vec<f64> = linspace(-1.0, 5.0, 100).into_iter().collect();
    let y_range: Vec<f64> = linspace(-2.0, 4.0, 100).into_iter().collect();

    let levels = vec![0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
    let colors = vec![&CYAN, &MAGENTA, &YELLOW, &RED, &BLUE, &GREEN];

    for (level, color) in levels.iter().zip(colors.iter()) {
        let mut contour_points = Vec::new();

        // Simple contour extraction (not perfect but works for this example)
        for i in 0..x_range.len() - 1 {
            for j in 0..y_range.len() - 1 {
                let x = x_range[i];
                let y = y_range[j];
                let z = objective_function(x, y);

                if (z - level).abs() < 0.1 {
                    contour_points.push((x, y));
                }
            }
        }

        if !contour_points.is_empty() {
            chart.draw_series(
                contour_points
                    .iter()
                    .map(|(x, y)| Circle::new((*x, *y), 1, color.filled())),
            )?;
        }
    }

    // Perform gradient descent again
    let mut current_pos = Array2::from_elem((1, 2), 0.0);
    current_pos[[0, 0]] = 0.0; // x
    current_pos[[0, 1]] = 3.0; // y

    let learning_rate = 0.1;
    let num_iterations = 50;
    let mut path = Vec::new();

    for _i in 0..num_iterations {
        let x = current_pos[[0, 0]];
        let y = current_pos[[0, 1]];
        path.push((x, y));

        let grad = numerical_gradient(objective_function_array, &current_pos);
        current_pos = &current_pos - &(grad * learning_rate);
    }
    path.push((current_pos[[0, 0]], current_pos[[0, 1]]));

    // Draw the gradient descent path
    chart
        .draw_series(LineSeries::new(path.iter().map(|(x, y)| (*x, *y)), &BLACK))?
        .label("Gradient Descent Path")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLACK));

    // Draw path points
    chart.draw_series(
        path.iter()
            .map(|(x, y)| Circle::new((*x, *y), 2, BLACK.filled())),
    )?;

    // Mark important points
    chart
        .draw_series(std::iter::once(Circle::new(path[0], 5, RED.filled())))?
        .label("Start")
        .legend(|(x, y)| Circle::new((x + 5, y), 3, RED.filled()));

    chart
        .draw_series(std::iter::once(Circle::new(
            path[path.len() - 1],
            5,
            GREEN.filled(),
        )))?
        .label("End")
        .legend(|(x, y)| Circle::new((x + 5, y), 3, GREEN.filled()));

    chart
        .draw_series(std::iter::once(Circle::new((2.0, 1.0), 6, BLUE.filled())))?
        .label("True Minimum")
        .legend(|(x, y)| Circle::new((x + 5, y), 3, BLUE.filled()));

    chart.configure_series_labels().draw()?;
    root.present()?;
    println!("Gradient descent contour plot saved to output/gradient_descent_contour.png");
    Ok(())
}
