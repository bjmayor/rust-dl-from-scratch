// examples/plot_activation_functions.rs
use ndarray::{Array2, linspace};
use plotters::prelude::*;
use rust_dl_from_scratch::chapter02::activation::{sigmoid, softmax};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Plotting activation functions...");

    // Create output directory if it doesn't exist
    std::fs::create_dir_all("output")?;

    plot_sigmoid()?;
    plot_softmax()?;
    plot_relu_and_tanh()?;

    println!("All plots saved to output/ directory");
    Ok(())
}

fn plot_sigmoid() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("output/sigmoid.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Sigmoid Activation Function", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(-10f64..10f64, 0f64..1f64)?;

    chart
        .configure_mesh()
        .x_desc("x")
        .y_desc("sigmoid(x)")
        .draw()?;

    // Generate sigmoid data
    let x_vals: Vec<f64> = linspace(-10.0, 10.0, 1000).into_iter().collect();
    let y_vals: Vec<f64> = x_vals
        .iter()
        .map(|&x| {
            let input = Array2::from_elem((1, 1), x);
            let output = sigmoid(&input);
            output[[0, 0]]
        })
        .collect();

    chart
        .draw_series(LineSeries::new(
            x_vals.iter().zip(y_vals.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))?
        .label("sigmoid(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));

    chart.configure_series_labels().draw()?;
    root.present()?;
    println!("Sigmoid plot saved to output/sigmoid.png");
    Ok(())
}

fn plot_softmax() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("output/softmax.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Softmax Function (3 classes)", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(-5f64..5f64, 0f64..1f64)?;

    chart
        .configure_mesh()
        .x_desc("x1 (x2=0, x3=0)")
        .y_desc("softmax probability")
        .draw()?;

    // Generate softmax data
    let x_vals: Vec<f64> = linspace(-5.0, 5.0, 200).into_iter().collect();
    let mut y1_vals = Vec::new();
    let mut y2_vals = Vec::new();
    let mut y3_vals = Vec::new();

    for &x1 in &x_vals {
        let input = Array2::from_elem((1, 3), 0.0);
        let mut input_mut = input.clone();
        input_mut[[0, 0]] = x1;
        input_mut[[0, 1]] = 0.0;
        input_mut[[0, 2]] = 0.0;

        let output = softmax(&input_mut);
        y1_vals.push(output[[0, 0]]);
        y2_vals.push(output[[0, 1]]);
        y3_vals.push(output[[0, 2]]);
    }

    chart
        .draw_series(LineSeries::new(
            x_vals.iter().zip(y1_vals.iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))?
        .label("class 1")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            x_vals.iter().zip(y2_vals.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))?
        .label("class 2")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(
            x_vals.iter().zip(y3_vals.iter()).map(|(&x, &y)| (x, y)),
            &GREEN,
        ))?
        .label("class 3")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &GREEN));

    chart.configure_series_labels().draw()?;
    root.present()?;
    println!("Softmax plot saved to output/softmax.png");
    Ok(())
}

fn plot_relu_and_tanh() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("output/relu_tanh.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("ReLU and Tanh Activation Functions", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(-5f64..5f64, -1f64..5f64)?;

    chart.configure_mesh().x_desc("x").y_desc("f(x)").draw()?;

    // Generate data
    let x_vals: Vec<f64> = linspace(-5.0, 5.0, 1000).into_iter().collect();

    // ReLU function
    let relu_vals: Vec<f64> = x_vals.iter().map(|&x| x.max(0.0)).collect();

    // Tanh function
    let tanh_vals: Vec<f64> = x_vals.iter().map(|&x| x.tanh()).collect();

    chart
        .draw_series(LineSeries::new(
            x_vals.iter().zip(relu_vals.iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))?
        .label("ReLU(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            x_vals.iter().zip(tanh_vals.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))?
        .label("tanh(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));

    chart.configure_series_labels().draw()?;
    root.present()?;
    println!("ReLU and Tanh plot saved to output/relu_tanh.png");
    Ok(())
}
