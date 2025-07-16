# Rust Deep Learning Visualization Examples 🎨

This directory contains examples demonstrating how to create publication-quality plots in Rust using the `plotters` library, serving as an equivalent to Python's matplotlib for deep learning visualizations.

## Overview

These examples showcase various visualization techniques commonly used in machine learning and deep learning projects:

- **Activation Functions**: Plot sigmoid, ReLU, tanh, and softmax functions
- **Training Progress**: Visualize loss curves during neural network training
- **Gradient Descent**: Show optimization paths and convergence
- **Loss Landscapes**: Create heatmaps and contour plots of loss functions
- **Data Visualization**: Scatter plots, classification datasets, and regression data

## Prerequisites

Make sure you have the `plotters` dependency in your `Cargo.toml`:

```toml
[dependencies]
plotters = "0.3"
```

## Running Examples

### Quick Start - Run All Examples

```bash
# Run the comprehensive example that generates all plots
cargo run --example all_plots
```

This will create an `output/` directory with all generated visualizations.

### Individual Examples

Run specific examples to focus on particular visualization types:

```bash
# Plot activation functions (sigmoid, ReLU, tanh, softmax)
cargo run --example plot_activation_functions

# Show neural network training progress
cargo run --example plot_training_loss

# Visualize gradient descent optimization
cargo run --example plot_gradient_descent

# Create loss function surface plots
cargo run --example plot_loss_surface

# Generate various data visualization examples
cargo run --example plot_data_visualization
```

## Generated Files

After running the examples, you'll find these files in the `output/` directory:

### Activation Functions
- `activation_functions.png` - Side-by-side comparison of activation functions
- `sigmoid.png` - Detailed sigmoid function plot
- `softmax.png` - Softmax function for multi-class classification
- `relu_tanh.png` - ReLU and Tanh comparison

### Training and Optimization
- `training_loss_demo.png` - Neural network training progress
- `training_loss.png` - Detailed loss curve with actual training
- `gradient_descent_demo.png` - Optimization path visualization
- `gradient_descent_2d.png` - 2D gradient descent path
- `gradient_descent_contour.png` - Gradient descent with contour lines

### Loss Landscapes
- `loss_heatmap_demo.png` - Loss function heatmap
- `loss_surface_3d.png` - 3D-style loss surface projection
- `loss_heatmap.png` - High-resolution loss landscape

### Data Visualization
- `scatter_plot.png` - Basic scatter plot example
- `classification_data.png` - Binary classification dataset (two moons)
- `regression_data.png` - Regression data with polynomial fit
- `multiple_datasets.png` - Four different data patterns

### Comparisons
- `comparison_demo.png` - Before/after training comparisons

## Key Features Demonstrated

### 1. Matplotlib-Style API
```rust
use plotters::prelude::*;

let root = BitMapBackend::new("output/plot.png", (800, 600)).into_drawing_area();
let mut chart = ChartBuilder::on(&root)
    .caption("My Plot", ("sans-serif", 40))
    .x_label_area_size(50)
    .y_label_area_size(50)
    .build_cartesian_2d(-10f64..10f64, -1f64..1f64)?;
```

### 2. Line Plots and Scatter Plots
```rust
// Line plot
chart.draw_series(LineSeries::new(data_points, &BLUE))?;

// Scatter plot
chart.draw_series(
    data.iter().map(|(x, y)| Circle::new((*x, *y), 3, RED.filled()))
)?;
```

### 3. Subplots and Multi-Panel Figures
```rust
let (left, right) = root.split_evenly((1, 2));
// Create separate charts on left and right panels
```

### 4. Heatmaps and Color Maps
```rust
let color = RGBColor(red_component, 0, blue_component);
Rectangle::new([(x, y), (x + dx, y + dy)], color.filled())
```

### 5. Legends and Annotations
```rust
chart.draw_series(series)?
    .label("Data Series")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));
```

## Rust vs Python Matplotlib Comparison

| Feature | Python Matplotlib | Rust Plotters |
|---------|------------------|---------------|
| Line plots | `plt.plot(x, y)` | `LineSeries::new(data, &BLUE)` |
| Scatter plots | `plt.scatter(x, y)` | `Circle::new((x, y), size, color)` |
| Subplots | `plt.subplot(2, 1, 1)` | `root.split_evenly((2, 1))` |
| Heatmaps | `plt.imshow(data)` | `Rectangle::new(coords, color)` |
| Legends | `plt.legend()` | `.legend()` + `configure_series_labels()` |
| Save figure | `plt.savefig()` | `root.present()` |

## Advanced Examples

### Real-time Training Visualization
The training examples show how to:
- Plot loss curves during actual neural network training
- Visualize gradient descent optimization paths
- Create before/after comparison plots

### Custom Color Maps
Examples demonstrate:
- Creating custom color gradients for heatmaps
- Mapping data values to colors
- Using different color schemes for different data types

### Mathematical Function Plotting
See how to:
- Plot activation functions with high precision
- Create contour plots of 2D functions
- Visualize mathematical surfaces

## Tips for Deep Learning Visualizations

1. **Real-time Updates**: For training monitoring, save plots at intervals
2. **Color Choice**: Use colorblind-friendly palettes
3. **Resolution**: Higher resolution for publication-quality figures
4. **Multiple Formats**: plotters supports PNG, SVG, and more
5. **Performance**: Efficient for large datasets compared to other solutions

## Extending the Examples

To create your own visualizations:

1. **Copy** an existing example as a template
2. **Modify** the data generation or plotting logic
3. **Customize** colors, styles, and layout
4. **Add** your specific deep learning metrics or data

## Output Quality

All generated plots are:
- **High Resolution**: 800x600 or higher
- **Publication Ready**: Vector graphics support via SVG
- **Customizable**: Easy to modify colors, fonts, and layout
- **Fast**: Efficient rendering even for large datasets

## Performance Notes

The `plotters` library is highly optimized and can handle:
- Large datasets (100k+ points)
- Real-time plotting
- Multiple output formats
- Complex multi-panel figures

Perfect for deep learning projects where you need fast, reliable visualizations! 🚀