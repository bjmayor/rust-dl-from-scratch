// examples/plot_data_visualization.rs
use plotters::prelude::*;
use rand::{Rng, thread_rng};
use rand_distr::{Distribution, Normal, Uniform};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating data visualization examples...");

    // Create output directory if it doesn't exist
    std::fs::create_dir_all("output")?;

    plot_scatter_data()?;
    plot_classification_data()?;
    plot_regression_data()?;
    plot_multiple_datasets()?;

    println!("Data visualization plots saved to output/ directory");
    Ok(())
}

fn plot_scatter_data() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("output/scatter_plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Scatter Plot Example", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(-3f64..3f64, -3f64..3f64)?;

    chart.configure_mesh().x_desc("X").y_desc("Y").draw()?;

    // Generate random data
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    let data1: Vec<(f64, f64)> = (0..100)
        .map(|_| {
            let x = normal.sample(&mut rng);
            let y = normal.sample(&mut rng);
            (x, y)
        })
        .collect();

    let data2: Vec<(f64, f64)> = (0..100)
        .map(|_| {
            let x = normal.sample(&mut rng) + 1.5;
            let y = normal.sample(&mut rng) + 1.5;
            (x, y)
        })
        .collect();

    // Plot first dataset
    chart
        .draw_series(
            data1
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 3, BLUE.filled())),
        )?
        .label("Dataset 1")
        .legend(|(x, y)| Circle::new((x + 5, y), 3, BLUE.filled()));

    // Plot second dataset
    chart
        .draw_series(
            data2
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 3, RED.filled())),
        )?
        .label("Dataset 2")
        .legend(|(x, y)| Circle::new((x + 5, y), 3, RED.filled()));

    chart.configure_series_labels().draw()?;
    root.present()?;
    println!("Scatter plot saved to output/scatter_plot.png");
    Ok(())
}

fn plot_classification_data() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("output/classification_data.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Binary Classification Dataset", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(-4f64..4f64, -4f64..4f64)?;

    chart
        .configure_mesh()
        .x_desc("Feature 1")
        .y_desc("Feature 2")
        .draw()?;

    // Generate classification data (two moons pattern)
    let mut rng = thread_rng();
    let noise = Normal::new(0.0, 0.1).unwrap();

    let mut class_0 = Vec::new();
    let mut class_1 = Vec::new();

    // Generate first moon
    for i in 0..100 {
        let t = std::f64::consts::PI * (i as f64) / 100.0;
        let x = 2.0 * t.cos() + noise.sample(&mut rng);
        let y = t.sin() + noise.sample(&mut rng);
        class_0.push((x, y));
    }

    // Generate second moon
    for i in 0..100 {
        let t = std::f64::consts::PI * (i as f64) / 100.0;
        let x = 2.0 * (t + std::f64::consts::PI).cos() + 2.0 + noise.sample(&mut rng);
        let y = (t + std::f64::consts::PI).sin() - 0.5 + noise.sample(&mut rng);
        class_1.push((x, y));
    }

    // Plot class 0
    chart
        .draw_series(
            class_0
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 4, BLUE.filled())),
        )?
        .label("Class 0")
        .legend(|(x, y)| Circle::new((x + 5, y), 4, BLUE.filled()));

    // Plot class 1
    chart
        .draw_series(
            class_1
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 4, RED.filled())),
        )?
        .label("Class 1")
        .legend(|(x, y)| Circle::new((x + 5, y), 4, RED.filled()));

    chart.configure_series_labels().draw()?;
    root.present()?;
    println!("Classification data plot saved to output/classification_data.png");
    Ok(())
}

fn plot_regression_data() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("output/regression_data.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Regression Dataset with Polynomial Fit", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(-2f64..2f64, -2f64..6f64)?;

    chart.configure_mesh().x_desc("X").y_desc("Y").draw()?;

    // Generate regression data
    let mut rng = thread_rng();
    let noise = Normal::new(0.0, 0.3).unwrap();
    let uniform = Uniform::new(-2.0, 2.0).unwrap();

    let data: Vec<(f64, f64)> = (0..200)
        .map(|_| {
            let x = uniform.sample(&mut rng);
            let y = x * x + 0.5 * x + noise.sample(&mut rng); // Quadratic with noise
            (x, y)
        })
        .collect();

    // Plot data points
    chart
        .draw_series(
            data.iter()
                .map(|(x, y)| Circle::new((*x, *y), 2, BLUE.filled())),
        )?
        .label("Data Points")
        .legend(|(x, y)| Circle::new((x + 5, y), 2, BLUE.filled()));

    // Plot true function
    let true_func: Vec<(f64, f64)> = (-200..=200)
        .map(|i| {
            let x = (i as f64) * 0.02;
            let y = x * x + 0.5 * x;
            (x, y)
        })
        .collect();

    chart
        .draw_series(LineSeries::new(true_func, &RED))?
        .label("True Function: y = x² + 0.5x")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));

    chart.configure_series_labels().draw()?;
    root.present()?;
    println!("Regression data plot saved to output/regression_data.png");
    Ok(())
}

fn plot_multiple_datasets() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("output/multiple_datasets.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    // Split into 4 subplots
    let areas = root.split_evenly((2, 1));
    let upper = &areas[0];
    let lower = &areas[1];
    let upper_areas = upper.split_evenly((1, 2));
    let upper_left = &upper_areas[0];
    let upper_right = &upper_areas[1];
    let lower_areas = lower.split_evenly((1, 2));
    let lower_left = &lower_areas[0];
    let lower_right = &lower_areas[1];

    // Subplot 1: Normal distribution
    {
        let mut chart = ChartBuilder::on(&upper_left)
            .caption("Normal Distribution", ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(-3f64..3f64, -3f64..3f64)?;

        chart.configure_mesh().draw()?;

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let data: Vec<(f64, f64)> = (0..100)
            .map(|_| (normal.sample(&mut rng), normal.sample(&mut rng)))
            .collect();

        chart.draw_series(
            data.iter()
                .map(|(x, y)| Circle::new((*x, *y), 2, BLUE.filled())),
        )?;
    }

    // Subplot 2: Uniform distribution
    {
        let mut chart = ChartBuilder::on(&upper_right)
            .caption("Uniform Distribution", ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(-1f64..1f64, -1f64..1f64)?;

        chart.configure_mesh().draw()?;

        let mut rng = thread_rng();
        let uniform = Uniform::new(-1.0, 1.0).unwrap();

        let data: Vec<(f64, f64)> = (0..100)
            .map(|_| (uniform.sample(&mut rng), uniform.sample(&mut rng)))
            .collect();

        chart.draw_series(
            data.iter()
                .map(|(x, y)| Circle::new((*x, *y), 2, RED.filled())),
        )?;
    }

    // Subplot 3: Circular pattern
    {
        let mut chart = ChartBuilder::on(&lower_left)
            .caption("Circular Pattern", ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(-2f64..2f64, -2f64..2f64)?;

        chart.configure_mesh().draw()?;

        let mut rng = thread_rng();
        let noise = Normal::new(0.0, 0.1).unwrap();

        let data: Vec<(f64, f64)> = (0..100)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * (i as f64) / 100.0;
                let radius = 1.0 + noise.sample(&mut rng);
                let x = radius * angle.cos();
                let y = radius * angle.sin();
                (x, y)
            })
            .collect();

        chart.draw_series(
            data.iter()
                .map(|(x, y)| Circle::new((*x, *y), 2, GREEN.filled())),
        )?;
    }

    // Subplot 4: Spiral pattern
    {
        let mut chart = ChartBuilder::on(&lower_right)
            .caption("Spiral Pattern", ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(-3f64..3f64, -3f64..3f64)?;

        chart.configure_mesh().draw()?;

        let mut rng = thread_rng();
        let noise = Normal::new(0.0, 0.05).unwrap();

        let data: Vec<(f64, f64)> = (0..200)
            .map(|i| {
                let t = (i as f64) * 0.1;
                let radius = 0.1 * t;
                let x = radius * t.cos() + noise.sample(&mut rng);
                let y = radius * t.sin() + noise.sample(&mut rng);
                (x, y)
            })
            .collect();

        chart.draw_series(
            data.iter()
                .map(|(x, y)| Circle::new((*x, *y), 2, MAGENTA.filled())),
        )?;
    }

    root.present()?;
    println!("Multiple datasets plot saved to output/multiple_datasets.png");
    Ok(())
}
