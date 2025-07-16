mod chapter01;
use chapter01::cli::interactive_mode;
mod chapter02;

use chapter02::train_simple::train_example;

fn main() {
    train_example();
}
