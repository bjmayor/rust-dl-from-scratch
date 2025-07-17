# MNIST Dataset Loading Guide

This guide explains how to use the MNIST dataset loading functionality in the rust-dl-from-scratch project.

## Overview

The MNIST dataset is a collection of 70,000 handwritten digit images (0-9) commonly used for training machine learning models. Our implementation provides easy-to-use functions for loading and preprocessing this data.

## Features

- **Automatic Download**: Downloads MNIST data files automatically if not present
- **Multiple Loading Options**: Various convenience functions for different use cases
- **Preprocessing**: Built-in normalization and one-hot encoding
- **Error Handling**: Comprehensive error handling for network and file issues
- **Memory Efficient**: Uses ndarray for efficient numerical operations

## Quick Start

Add the following dependencies to your `Cargo.toml`:

```toml
[dependencies]
ndarray = "0.16"
flate2 = "1.0"
byteorder = "1.5"
reqwest = { version = "0.11", features = ["blocking"] }
```

### Basic Usage

```rust
use rust_dl_from_scratch::datasets::MnistDataset;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the complete dataset
    let mnist = MnistDataset::load()?;
    
    println!("Training samples: {}", mnist.train_size());
    println!("Test samples: {}", mnist.test_size());
    
    Ok(())
}
```

## Loading Functions

### 1. Complete Dataset

Load the full MNIST dataset with all features:

```rust
let mut mnist = MnistDataset::load()?;
mnist.normalize(); // Convert pixels to [0, 1] range

// Access data
let train_images = &mnist.train_images; // Shape: [60000, 784]
let train_labels = &mnist.train_labels; // Shape: [60000]
let test_images = &mnist.test_images;   // Shape: [10000, 784]
let test_labels = &mnist.test_labels;   // Shape: [10000]
```

### 2. Quick Loading with Normalization

For most use cases, you'll want normalized data:

```rust
// Load training data only
let (train_images, train_labels) = MnistDataset::load_train_normalized()?;

// Load test data only
let (test_images, test_labels) = MnistDataset::load_test_normalized()?;
```

### 3. One-Hot Encoded Labels

For neural networks, you often need one-hot encoded labels:

```rust
let (train_x, train_y, test_x, test_y) = MnistDataset::load_one_hot()?;

// train_y and test_y are now [samples, 10] arrays
// where each row is a one-hot vector (e.g., [0,0,0,1,0,0,0,0,0,0] for digit 3)
```

### 4. Small Subset for Testing

For quick experiments or debugging:

```rust
let (images, labels) = MnistDataset::load_small_subset()?; // First 1000 samples
```

## Data Format

### Images
- **Shape**: `[samples, 784]` (28×28 pixels flattened)
- **Type**: `Array2<f32>`
- **Range**: `[0, 255]` (raw) or `[0, 1]` (normalized)

### Labels
- **Shape**: `[samples]` for regular labels, `[samples, 10]` for one-hot
- **Type**: `Array1<u8>` for regular, `Array2<f32>` for one-hot
- **Range**: `0-9` (digits)

## Batch Processing

Get batches of data for training:

```rust
let mnist = MnistDataset::load()?;

// Get specific samples by index
let indices = vec![0, 1, 2, 3, 4];
let (batch_images, batch_labels) = mnist.get_train_batch(&indices);
```

## Data Preprocessing

### Normalization

Convert pixel values from [0, 255] to [0, 1]:

```rust
let mut mnist = MnistDataset::load()?;
mnist.normalize();
```

### One-Hot Encoding

Convert integer labels to one-hot vectors:

```rust
use ndarray::Array1;

let labels = Array1::from_vec(vec![0, 1, 2]);
let one_hot = mnist.labels_to_one_hot(&labels);
// Result: [[1,0,0,0,0,0,0,0,0,0],
//          [0,1,0,0,0,0,0,0,0,0],
//          [0,0,1,0,0,0,0,0,0,0]]
```

## Error Handling

The library provides comprehensive error handling:

```rust
use rust_dl_from_scratch::datasets::MnistError;

match MnistDataset::load() {
    Ok(dataset) => {
        // Use dataset
    }
    Err(MnistError::IoError(e)) => {
        eprintln!("File system error: {}", e);
    }
    Err(MnistError::HttpError(e)) => {
        eprintln!("Network error: {}", e);
    }
    Err(MnistError::InvalidMagicNumber) => {
        eprintln!("Corrupted MNIST file");
    }
    Err(MnistError::InvalidDimensions) => {
        eprintln!("Unexpected data dimensions");
    }
}
```

## File Storage

Downloaded files are stored in `data/mnist/`:
- `train-images-idx3-ubyte.gz` (60,000 training images)
- `train-labels-idx1-ubyte.gz` (60,000 training labels)
- `t10k-images-idx3-ubyte.gz` (10,000 test images)
- `t10k-labels-idx1-ubyte.gz` (10,000 test labels)

Files are only downloaded once and reused on subsequent runs.

## Examples

Run the included examples:

```bash
# Complete dataset demonstration
cargo run --example mnist_example

# Simple usage patterns
cargo run --example simple_mnist
```

## Performance Tips

1. **Use small subsets** for initial development and debugging
2. **Load data once** and reuse rather than reloading
3. **Normalize data** early in your pipeline
4. **Use batching** for memory-efficient training

## Testing

Run the tests to verify everything works:

```bash
# Run all tests
cargo test

# Run MNIST-specific tests (requires internet)
cargo test test_mnist_loading -- --ignored
```

## Data Source

The MNIST data is downloaded from:
- `https://ossci-datasets.s3.amazonaws.com/mnist/`

This is a reliable mirror of the original MNIST dataset from Yann LeCun's website.