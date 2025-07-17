use ndarray::Array1;
use rust_dl_from_scratch::datasets::MnistDataset;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading MNIST dataset...");

    // Load the MNIST dataset
    let mut mnist = MnistDataset::load()?;

    // Print dataset information
    println!("Dataset loaded successfully!");
    println!("Training samples: {}", mnist.train_size());
    println!("Test samples: {}", mnist.test_size());
    println!("Image size: {} pixels (28x28)", mnist.image_size());

    // Normalize the data to [0, 1] range
    mnist.normalize();
    println!("Data normalized to [0, 1] range");

    // Show some statistics about the first training image
    let first_image = mnist.train_images.row(0);
    let first_label = mnist.train_labels[0];

    println!("\nFirst training sample:");
    println!("Label: {}", first_label);
    println!(
        "Min pixel value: {:.3}",
        first_image.iter().fold(f32::INFINITY, |a, &b| a.min(b))
    );
    println!(
        "Max pixel value: {:.3}",
        first_image.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!("Mean pixel value: {:.3}", first_image.mean().unwrap());

    // Show distribution of labels in training set
    let mut label_counts = vec![0; 10];
    for &label in mnist.train_labels.iter() {
        label_counts[label as usize] += 1;
    }

    println!("\nLabel distribution in training set:");
    for (digit, count) in label_counts.iter().enumerate() {
        println!("Digit {}: {} samples", digit, count);
    }

    // Demonstrate batch loading
    let batch_indices = vec![0, 1, 2, 3, 4];
    let (batch_images, batch_labels) = mnist.get_train_batch(&batch_indices);
    println!("\nLoaded batch of {} images", batch_images.nrows());
    println!("Batch labels: {:?}", batch_labels.to_vec());

    // Demonstrate one-hot encoding
    let sample_labels = Array1::from_vec(vec![0, 1, 2, 3, 4]);
    let one_hot = mnist.labels_to_one_hot(&sample_labels);
    println!("\nOne-hot encoding example:");
    println!("Original labels: {:?}", sample_labels.to_vec());
    println!("One-hot shape: {:?}", one_hot.shape());

    // Print first few rows of one-hot encoding
    for (i, row) in one_hot.rows().into_iter().enumerate().take(3) {
        println!("Label {}: {:?}", sample_labels[i], row.to_vec());
    }

    // Visualize a sample image in ASCII
    println!(
        "\nASCII visualization of first training image (label: {}):",
        first_label
    );
    visualize_image_ascii(&first_image.to_owned());

    Ok(())
}

/// Simple ASCII visualization of MNIST image
fn visualize_image_ascii(image: &Array1<f32>) {
    const WIDTH: usize = 28;
    const HEIGHT: usize = 28;

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let pixel_value = image[y * WIDTH + x];
            let char = if pixel_value > 0.8 {
                "██"
            } else if pixel_value > 0.6 {
                "▓▓"
            } else if pixel_value > 0.4 {
                "▒▒"
            } else if pixel_value > 0.2 {
                "░░"
            } else {
                "  "
            };
            print!("{}", char);
        }
        println!();
    }
}
