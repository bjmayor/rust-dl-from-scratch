// examples/plot_loss_surface.rs
use ndarray::{Array2, linspace};
use plotters::prelude::*;
use rust_dl_from_scratch::chapter02::loss::cross_entropy_error;
use rust_dl_from_scratch::chapter02::network::SimpleNet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Plotting loss function surface...");

    // Create output directory if it doesn't exist
    std::fs::create_dir_all("output")?;

    plot_loss_surface()?;
    plot_loss_heatmap()?;

    println!("Loss surface plots saved to output/ directory");
    Ok(())
}

fn plot_loss_surface() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("output/loss_surface_3d.png", (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    // Create sample data
    let x = Array2::from_elem((1, 2), 0.0);
    let mut x_mut = x.clone();
    x_mut[[0, 0]] = 0.6;
    x_mut[[0, 1]] = 0.9;

    let t = Array2::from_elem((1, 2), 0.0);
    let mut t_mut = t.clone();
    t_mut[[0, 0]] = 0.0;
    t_mut[[0, 1]] = 1.0;

    // Create a simple 3D-like visualization by varying two parameters
    let w1_range: Vec<f64> = linspace(-3.0, 3.0, 20).into_iter().collect();
    let w2_range: Vec<f64> = linspace(-3.0, 3.0, 20).into_iter().collect();

    let mut chart = ChartBuilder::on(&root)
        .caption("Loss Function Surface (2D projection)", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(-3f64..3f64, -3f64..3f64)?;

    chart
        .configure_mesh()
        .x_desc("Weight 1")
        .y_desc("Weight 2")
        .draw()?;

    // Calculate loss for different weight combinations
    let mut loss_data = Vec::new();
    for &w1 in w1_range.iter() {
        for &w2 in w2_range.iter() {
            // Create a simple network with fixed architecture
            let mut net = SimpleNet::new(2, 3, 2);

            // Modify one weight to see how loss changes
            net.w1[[0, 0]] = w1;
            net.w2[[0, 0]] = w2;

            let y = net.predict(&x_mut);
            let loss = cross_entropy_error(&y, &t_mut);

            loss_data.push((w1, w2, loss));
        }
    }

    // Find min and max loss for color mapping
    let min_loss = loss_data
        .iter()
        .map(|(_, _, loss)| *loss)
        .fold(f64::INFINITY, f64::min);
    let max_loss = loss_data
        .iter()
        .map(|(_, _, loss)| *loss)
        .fold(0.0, f64::max);

    // Draw points with colors representing loss values
    chart.draw_series(loss_data.iter().map(|(w1, w2, loss)| {
        let normalized_loss = (loss - min_loss) / (max_loss - min_loss);
        let color_intensity = (255.0 * (1.0 - normalized_loss)) as u8;
        let color = RGBColor(255 - color_intensity, color_intensity, 0);
        Circle::new((*w1, *w2), 3, color.filled())
    }))?;

    root.present()?;
    println!("Loss surface plot saved to output/loss_surface_3d.png");
    Ok(())
}

fn plot_loss_heatmap() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("output/loss_heatmap.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Loss Function Heatmap", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(-3f64..3f64, -3f64..3f64)?;

    chart
        .configure_mesh()
        .x_desc("Weight 1")
        .y_desc("Weight 2")
        .draw()?;

    // Create sample data
    let x = Array2::from_elem((1, 2), 0.0);
    let mut x_mut = x.clone();
    x_mut[[0, 0]] = 0.6;
    x_mut[[0, 1]] = 0.9;

    let t = Array2::from_elem((1, 2), 0.0);
    let mut t_mut = t.clone();
    t_mut[[0, 0]] = 0.0;
    t_mut[[0, 1]] = 1.0;

    // Higher resolution for heatmap
    let resolution = 50;
    let w1_range: Vec<f64> = linspace(-3.0, 3.0, resolution).into_iter().collect();
    let w2_range: Vec<f64> = linspace(-3.0, 3.0, resolution).into_iter().collect();

    let mut loss_grid = Vec::new();

    for (i, &w1) in w1_range.iter().enumerate() {
        for (j, &w2) in w2_range.iter().enumerate() {
            let mut net = SimpleNet::new(2, 3, 2);
            net.w1[[0, 0]] = w1;
            net.w2[[0, 0]] = w2;

            let y = net.predict(&x_mut);
            let loss = cross_entropy_error(&y, &t_mut);

            loss_grid.push((i, j, w1, w2, loss));
        }
    }

    // Find min and max for normalization
    let min_loss = loss_grid
        .iter()
        .map(|(_, _, _, _, loss)| *loss)
        .fold(f64::INFINITY, f64::min);
    let max_loss = loss_grid
        .iter()
        .map(|(_, _, _, _, loss)| *loss)
        .fold(0.0, f64::max);

    // Create heatmap using rectangles
    let step_w1 = 6.0 / resolution as f64;
    let step_w2 = 6.0 / resolution as f64;

    chart.draw_series(loss_grid.iter().map(|(_, _, w1, w2, loss)| {
        let normalized_loss = (loss - min_loss) / (max_loss - min_loss);

        // Create a color gradient from blue (low loss) to red (high loss)
        let red_component = (255.0 * normalized_loss) as u8;
        let blue_component = (255.0 * (1.0 - normalized_loss)) as u8;
        let color = RGBColor(red_component, 0, blue_component);

        Rectangle::new([(*w1, *w2), (*w1 + step_w1, *w2 + step_w2)], color.filled())
    }))?;

    // Add contour lines for better visualization
    let contour_levels = vec![
        min_loss + 0.2 * (max_loss - min_loss),
        min_loss + 0.4 * (max_loss - min_loss),
        min_loss + 0.6 * (max_loss - min_loss),
        min_loss + 0.8 * (max_loss - min_loss),
    ];

    for level in contour_levels {
        let mut contour_points = Vec::new();

        for (_, _, w1, w2, loss) in &loss_grid {
            if (loss - level).abs() < 0.05 * (max_loss - min_loss) {
                contour_points.push((*w1, *w2));
            }
        }

        chart.draw_series(
            contour_points
                .iter()
                .map(|(w1, w2)| Circle::new((*w1, *w2), 1, BLACK.filled())),
        )?;
    }

    root.present()?;
    println!("Loss heatmap saved to output/loss_heatmap.png");
    Ok(())
}
