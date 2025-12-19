use ndarray::Array1;
use ndarray_linalg::Norm;
use rand::distr::Uniform;
use rand::{Rng, rng};

use crate::errors::OVSAError;

/// Generates a random dense vector of given size with values uniformly distributed between min and max.
/// # Arguments
/// * `dimension` - The size of the vector.
/// * `min` - The minimum value for the uniform distribution.
/// * `max` - The maximum value for the uniform distribution.
/// # Returns
/// A dense vector represented as `Array1<f32>`.
pub fn random_uniform(dimension: usize, min: f32, max: f32) -> Result<Array1<f32>, OVSAError> {
    if dimension == 0 {
        return Err(OVSAError::ZeroDimension);
    }

    let mut rng = rng();
    let uniform = Uniform::new(min, max).unwrap();

    Ok(
        Array1::from(
        (&mut rng).sample_iter(&uniform).take(dimension).collect::<Vec<f32>>()
        )
    )
}


/// Computes the superposition (element-wise sum) of a slice of dense vectors.
/// # Arguments
/// * `array_vec` - A slice of dense vectors represented as `Array1<f32>`.
/// # Returns
/// A dense vector representing the superposition result.
pub fn superposition(array_vec: &[Array1<f32>]) -> Result<Array1<f32>, OVSAError> {
    if array_vec.is_empty() {
        return Err(OVSAError::EmptyVectorList);
    }

    let size = array_vec.get(0).expect("Input slice is empty").len();

    let mut result = Array1::<f32>::zeros(array_vec[0].len());
    // todo: optimize
    for array in array_vec {
        if array.len() != size {
            return Err(OVSAError::VectorSizeMismatch);
        }
        result += array;
    }

    Ok(result)
}


/// Computes the circular convolution of two dense vectors.
/// # Arguments
/// * `a` - The first dense vector.
/// * `b` - The second dense vector.
/// # Returns
/// A dense vector representing the circular convolution result.
pub fn circular_convolution(a: &Array1<f32>, b: &Array1<f32>) -> Array1<f32> {
    let n = a.len();
    let mut result = Array1::<f32>::zeros(n);

    // todo: optimize with matmul and slices
    for i in 0..n {
        for j in 0..n {
            let k = (i + j) % n;
            result[k] += a[i] * b[j];
        }
    }

    result
}



/// Computes the circular correlation of two dense vectors./// # Arguments
/// * `a` - The first dense vector.
/// * `b` - The second dense vector.
/// # Returns
/// A dense vector representing the circular correlation result.
pub fn circular_correlation(a: &Array1<f32>, b: &Array1<f32>) -> Array1<f32> {
    let n = a.len();
    let mut result = Array1::<f32>::zeros(n);
    for i in 0..n {
        for j in 0..n {
            let k = (i + n - j) % n;
            result[k] += a[i] * b[j];
        }
    }

    result
}


/// Performs a cyclic shift on a dense vector by a specified number of positions.
/// Positive values shift to the right, negative values shift to the left.
/// # Arguments
/// * `array` - The dense vector to be shifted.
/// * `shift_by` - The number of positions to shift.
/// # Returns
/// A new dense vector that has been cyclically shifted.
pub fn cyclic_shift(array: &Array1<f32>, shift_by: isize) -> Array1<f32> {
    let n = array.len() as isize;
    let mut result = Array1::<f32>::zeros(array.len());

    for old_index in 0..n {
        let new_index = (old_index + shift_by).rem_euclid(n);
        result[new_index as usize] = array[old_index as usize];
    }

    result
}


/// Computes the cosine similarity between two dense vectors.
/// # Arguments
/// * `a` - The first dense vector.
/// * `b` - The second dense vector.
/// # Returns
/// The cosine similarity as a f32 value between -1.0 and 1.0
pub fn similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must be of the same dimension for similarity computation.");

    a.dot(b) / (a.norm_l2() * b.norm_l2())
}