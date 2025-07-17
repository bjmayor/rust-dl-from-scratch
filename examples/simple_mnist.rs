use rust_dl_from_scratch::datasets::MnistDataset;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Simple MNIST Example");
    println!("===================");

    // Quick load of normalized training data
    println!("Loading training data...");
    let (train_images, train_labels) = MnistDataset::load_train_normalized()?;
    println!("✓ Loaded {} training samples", train_images.nrows());

    // Quick load of normalized test data
    println!("Loading test data...");
    let (test_images, test_labels) = MnistDataset::load_test_normalized()?;
    println!("✓ Loaded {} test samples", test_images.nrows());

    // Load small subset for quick experiments
    println!("Loading small subset...");
    let (small_images, small_labels) = MnistDataset::load_small_subset()?;
    println!(
        "✓ Loaded {} samples for quick testing",
        small_images.nrows()
    );

    // Load with one-hot encoded labels
    println!("Loading with one-hot encoding...");
    let (train_x, train_y, test_x, test_y) = MnistDataset::load_one_hot()?;
    println!(
        "✓ Training: {} samples with {} features",
        train_x.nrows(),
        train_x.ncols()
    );
    println!("✓ Training labels shape: {:?}", train_y.shape());
    println!(
        "✓ Test: {} samples with {} features",
        test_x.nrows(),
        test_x.ncols()
    );
    println!("✓ Test labels shape: {:?}", test_y.shape());

    // Show some sample statistics
    println!("\nDataset Statistics:");
    println!("- Image size: 28x28 = {} pixels", train_images.ncols());
    println!("- Pixel value range: [0.0, 1.0] (normalized)");
    println!("- Number of classes: 10 (digits 0-9)");

    // Show first sample info
    let first_image = train_images.row(0);
    let first_label = train_labels[0];
    println!("\nFirst training sample:");
    println!("- Label: {}", first_label);
    println!(
        "- Non-zero pixels: {}",
        first_image.iter().filter(|&&x| x > 0.0).count()
    );
    println!("- Mean pixel value: {:.3}", first_image.mean().unwrap());

    println!("\n✅ All MNIST loading functions work correctly!");

    Ok(())
}
