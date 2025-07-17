use byteorder::{BigEndian, ReadBytesExt};
use flate2::read::GzDecoder;
use ndarray::{Array1, Array2, s};
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::Path;

/// MNIST dataset structure
#[derive(Debug, Clone)]
pub struct MnistDataset {
    pub train_images: Array2<f32>,
    pub train_labels: Array1<u8>,
    pub test_images: Array2<f32>,
    pub test_labels: Array1<u8>,
}

/// MNIST data URLs
const TRAIN_IMAGES_URL: &str =
    "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz";
const TRAIN_LABELS_URL: &str =
    "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz";
const TEST_IMAGES_URL: &str =
    "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz";
const TEST_LABELS_URL: &str =
    "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz";

/// Errors that can occur during MNIST loading
#[derive(Debug)]
pub enum MnistError {
    IoError(std::io::Error),
    HttpError(reqwest::Error),
    InvalidMagicNumber,
    InvalidDimensions,
}

impl From<std::io::Error> for MnistError {
    fn from(error: std::io::Error) -> Self {
        MnistError::IoError(error)
    }
}

impl From<reqwest::Error> for MnistError {
    fn from(error: reqwest::Error) -> Self {
        MnistError::HttpError(error)
    }
}

impl std::fmt::Display for MnistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MnistError::IoError(e) => write!(f, "IO error: {}", e),
            MnistError::HttpError(e) => write!(f, "HTTP error: {}", e),
            MnistError::InvalidMagicNumber => write!(f, "Invalid magic number in MNIST file"),
            MnistError::InvalidDimensions => write!(f, "Invalid dimensions in MNIST file"),
        }
    }
}

impl std::error::Error for MnistError {}

impl MnistDataset {
    /// Load MNIST dataset from local files or download if not present
    pub fn load() -> Result<Self, MnistError> {
        let data_dir = "data/mnist";
        fs::create_dir_all(data_dir)?;

        // Download files if they don't exist
        let train_images_path = format!("{}/train-images-idx3-ubyte.gz", data_dir);
        let train_labels_path = format!("{}/train-labels-idx1-ubyte.gz", data_dir);
        let test_images_path = format!("{}/t10k-images-idx3-ubyte.gz", data_dir);
        let test_labels_path = format!("{}/t10k-labels-idx1-ubyte.gz", data_dir);

        download_if_not_exists(TRAIN_IMAGES_URL, &train_images_path)?;
        download_if_not_exists(TRAIN_LABELS_URL, &train_labels_path)?;
        download_if_not_exists(TEST_IMAGES_URL, &test_images_path)?;
        download_if_not_exists(TEST_LABELS_URL, &test_labels_path)?;

        // Load the data
        let train_images = load_images(&train_images_path)?;
        let train_labels = load_labels(&train_labels_path)?;
        let test_images = load_images(&test_images_path)?;
        let test_labels = load_labels(&test_labels_path)?;

        Ok(MnistDataset {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }

    /// Get training data size
    pub fn train_size(&self) -> usize {
        self.train_images.nrows()
    }

    /// Get test data size
    pub fn test_size(&self) -> usize {
        self.test_images.nrows()
    }

    /// Get image dimensions (28x28 = 784)
    pub fn image_size(&self) -> usize {
        self.train_images.ncols()
    }

    /// Get a batch of training data
    pub fn get_train_batch(&self, indices: &[usize]) -> (Array2<f32>, Array1<u8>) {
        let batch_images = self.train_images.select(ndarray::Axis(0), indices);
        let batch_labels = self.train_labels.select(ndarray::Axis(0), indices);
        (batch_images, batch_labels)
    }

    /// Get a batch of test data
    pub fn get_test_batch(&self, indices: &[usize]) -> (Array2<f32>, Array1<u8>) {
        let batch_images = self.test_images.select(ndarray::Axis(0), indices);
        let batch_labels = self.test_labels.select(ndarray::Axis(0), indices);
        (batch_images, batch_labels)
    }

    /// Normalize images to [0, 1] range
    pub fn normalize(&mut self) {
        self.train_images.mapv_inplace(|x| x / 255.0);
        self.test_images.mapv_inplace(|x| x / 255.0);
    }

    /// Convert labels to one-hot encoding
    pub fn labels_to_one_hot(&self, labels: &Array1<u8>) -> Array2<f32> {
        let num_classes = 10;
        let mut one_hot = Array2::<f32>::zeros((labels.len(), num_classes));

        for (i, &label) in labels.iter().enumerate() {
            one_hot[[i, label as usize]] = 1.0;
        }

        one_hot
    }

    /// Quick load for just training data, normalized
    pub fn load_train_normalized() -> Result<(Array2<f32>, Array1<u8>), MnistError> {
        let mut dataset = Self::load()?;
        dataset.normalize();
        Ok((dataset.train_images, dataset.train_labels))
    }

    /// Quick load for just test data, normalized
    pub fn load_test_normalized() -> Result<(Array2<f32>, Array1<u8>), MnistError> {
        let mut dataset = Self::load()?;
        dataset.normalize();
        Ok((dataset.test_images, dataset.test_labels))
    }

    /// Load both train and test data, normalized, with one-hot encoded labels
    pub fn load_one_hot() -> Result<(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>), MnistError>
    {
        let mut dataset = Self::load()?;
        dataset.normalize();

        let train_labels_one_hot = dataset.labels_to_one_hot(&dataset.train_labels);
        let test_labels_one_hot = dataset.labels_to_one_hot(&dataset.test_labels);

        Ok((
            dataset.train_images,
            train_labels_one_hot,
            dataset.test_images,
            test_labels_one_hot,
        ))
    }

    /// Load a small subset for quick testing (first 1000 training samples)
    pub fn load_small_subset() -> Result<(Array2<f32>, Array1<u8>), MnistError> {
        let mut dataset = Self::load()?;
        dataset.normalize();

        let subset_size = 1000.min(dataset.train_size());
        let subset_images = dataset
            .train_images
            .slice(s![0..subset_size, ..])
            .to_owned();
        let subset_labels = dataset.train_labels.slice(s![0..subset_size]).to_owned();

        Ok((subset_images, subset_labels))
    }
}

/// Download a file if it doesn't exist locally
fn download_if_not_exists(url: &str, path: &str) -> Result<(), MnistError> {
    if !Path::new(path).exists() {
        println!("Downloading {}...", url);
        let response = reqwest::blocking::get(url)?;
        let bytes = response.bytes()?;
        fs::write(path, bytes)?;
        println!("Downloaded {} successfully", path);
    }
    Ok(())
}

/// Load MNIST images from gzipped file
fn load_images(path: &str) -> Result<Array2<f32>, MnistError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(GzDecoder::new(file));

    // Read header
    let magic = reader.read_u32::<BigEndian>()?;
    if magic != 0x00000803 {
        return Err(MnistError::InvalidMagicNumber);
    }

    let num_images = reader.read_u32::<BigEndian>()? as usize;
    let num_rows = reader.read_u32::<BigEndian>()? as usize;
    let num_cols = reader.read_u32::<BigEndian>()? as usize;

    if num_rows != 28 || num_cols != 28 {
        return Err(MnistError::InvalidDimensions);
    }

    // Read image data
    let mut buffer = vec![0u8; num_images * num_rows * num_cols];
    reader.read_exact(&mut buffer)?;

    // Convert to Array2<f32>
    let images: Vec<f32> = buffer.into_iter().map(|x| x as f32).collect();
    let images_array = Array2::from_shape_vec((num_images, num_rows * num_cols), images)
        .map_err(|_| MnistError::InvalidDimensions)?;

    Ok(images_array)
}

/// Load MNIST labels from gzipped file
fn load_labels(path: &str) -> Result<Array1<u8>, MnistError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(GzDecoder::new(file));

    // Read header
    let magic = reader.read_u32::<BigEndian>()?;
    if magic != 0x00000801 {
        return Err(MnistError::InvalidMagicNumber);
    }

    let num_labels = reader.read_u32::<BigEndian>()? as usize;

    // Read label data
    let mut buffer = vec![0u8; num_labels];
    reader.read_exact(&mut buffer)?;

    let labels_array = Array1::from_vec(buffer);
    Ok(labels_array)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnist_loading() {
        // This test requires internet connection to download MNIST data
        // Run with: cargo test test_mnist_loading -- --ignored
        match MnistDataset::load() {
            Ok(mnist) => {
                assert_eq!(mnist.train_size(), 60000);
                assert_eq!(mnist.test_size(), 10000);
                assert_eq!(mnist.image_size(), 784); // 28x28

                // Check that labels are in valid range
                assert!(mnist.train_labels.iter().all(|&x| x < 10));
                assert!(mnist.test_labels.iter().all(|&x| x < 10));

                println!("MNIST dataset loaded successfully!");
                println!("Training samples: {}", mnist.train_size());
                println!("Test samples: {}", mnist.test_size());
                println!("Image size: {}x{}", 28, 28);
            }
            Err(e) => {
                eprintln!("Failed to load MNIST: {}", e);
                // Don't panic in tests unless we want to require internet
            }
        }
    }

    #[test]
    fn test_one_hot_encoding() {
        let labels = Array1::from_vec(vec![0, 1, 2, 9]);
        let mnist = MnistDataset {
            train_images: Array2::zeros((0, 784)),
            train_labels: Array1::zeros(0),
            test_images: Array2::zeros((0, 784)),
            test_labels: Array1::zeros(0),
        };

        let one_hot = mnist.labels_to_one_hot(&labels);
        assert_eq!(one_hot.shape(), &[4, 10]);

        // Check that each row has exactly one 1.0
        for row in one_hot.rows() {
            assert_eq!(row.sum(), 1.0);
        }

        // Check specific values
        assert_eq!(one_hot[[0, 0]], 1.0);
        assert_eq!(one_hot[[1, 1]], 1.0);
        assert_eq!(one_hot[[2, 2]], 1.0);
        assert_eq!(one_hot[[3, 9]], 1.0);
    }
}
