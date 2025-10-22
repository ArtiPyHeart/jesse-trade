//! # CWT (Continuous Wavelet Transform) Module
//!
//! High-performance Rust implementation of Continuous Wavelet Transform
//! with numerical alignment to PyWavelets library.

pub mod wavelets;
pub mod utils;
pub mod core;
pub mod ffi;

// Re-export main functions
pub use ffi::cwt_py;
