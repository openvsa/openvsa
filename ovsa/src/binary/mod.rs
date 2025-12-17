

use sprs::CsVec;
use rand::seq::index::sample;
use rand::distr::Uniform;
use rand::rng;

/// Generates a sparse random binary vector of given size with a specified number of active (1) entries.
/// This could probably be optimized to use bit fields instead of u8 vectors.
/// # Arguments
/// * `dimension` - The size of the vector.
/// * `n_active` - The number of active (1) entries in the vector.
/// # Returns
/// A sparse binary vector represented as `CsVec<i8>`.
pub fn sparse_random(dimension: usize, n_active: usize) -> CsVec<i8> {

    let mut rng  = rng();
    let mut indices: Vec<usize> = sample(&mut rng, dimension, n_active).into_vec();
    indices.sort();
    let data: Vec<i8> = vec![1i8; n_active];

    CsVec::new(dimension, indices, data)
}

/// Creates a sparse binary vector from given indices of active (1) entries.
/// # Arguments
/// * `dimension` - The size of the vector.
/// * `indices` - A slice of indices where the entries are active (1).
/// # Returns
/// A sparse binary vector represented as `CsVec<i8>`.
pub fn from_indices(dimension: usize, indices: &[usize]) -> CsVec<i8> {
    let n_active = indices.len();
    let data: Vec<i8> = vec![1i8; n_active];

    CsVec::new_from_unsorted(dimension, indices.to_vec(), data).unwrap()
}

/// Computes the Hamming distance between two sparse binary vectors.
/// The Hamming distance is defined as the number of positions at which the corresponding entries are different.
/// # Arguments
/// * `vec1` - The first sparse binary vector.
/// * `vec2` - The second sparse binary vector.
/// # Returns
/// The Hamming distance as a usize.
pub fn hamming_distance(vec1: &CsVec<i8>, vec2: &CsVec<i8>) -> usize {
    assert_eq!(vec1.dim(), vec2.dim(), "Vectors must be of the same dimension to compute Hamming distance.");

    let bound_vec = xor(vec1, vec2);
    bound_vec.nnz()
}

/// Computes the consensus sum of a slice of sparse binary vectors.
/// The consensus sum is determined by taking the majority value at each index across all vectors.
/// # Arguments
/// * `vectors` - A slice of sparse binary vectors represented as `CsVec<i8>`.
/// # Returns
/// A sparse binary vector representing the consensus sum.
pub fn consensus_sum(vectors: &[CsVec<i8>]) -> CsVec<i8> {
    // todo: optimize this to avoid using a full vector
    let size: usize = vectors[0].dim();
    let mut result_data: Vec<i16> = vec![0i16; size];

    for vec in vectors {
        let active_indices = vec.indices();
        for index in 0..size {
            if active_indices.contains(&index) {
                result_data[index] += 1;
            } else {
                result_data[index] -= 1;
            }
        }
    }

    let mut rng  = rng();
    let uniform = Uniform::new(0.0, 1.0).unwrap();

    fn set_active(value: i16, rng: &mut impl rand::Rng, uniform: &Uniform<f64>) -> bool {
        if value > 0 {
            true
        } else if value < 0 {
            false
        } else {
            rng.sample(uniform) > 0.5
        }
    }

    let mut indices: Vec<usize> = result_data.iter()
        .enumerate()
        .filter_map(|(index, &value)| if set_active(value, &mut rng, &uniform) { Some(index) } else { None })
        .collect();

    indices.sort();

    from_indices(size, &indices)
}

/// Computes the element-wise XOR of two sparse binary vectors.
/// # Arguments
/// * `vec1` - The first sparse binary vector.
/// * `vec2` - The second sparse binary vector.
/// # Returns
/// A sparse binary vector representing the XOR result.
pub fn xor(vec1: &CsVec<i8>, vec2: &CsVec<i8>) -> CsVec<i8> {
    assert_eq!(vec1.dim(), vec2.dim(), "Vectors must be of the same dimension for binding.");

    let size: usize = vec1.dim();
    // to simulate an XOR operation, we add the two vectors and keep only the entries where the sum is 1
    let result: CsVec<i8> = vec1 + vec2;
    let indices: Vec<usize> = result.iter()
        // XOR operation: 1 + 1 = 0, so we keep only entries with value 1
        .filter_map(|(index, &value)| if value == 1 { Some(index) } else { None })
        .collect();

    from_indices(size, &indices)
}

/// Performs a cyclic shift on a sparse binary vector.
/// Typically used for implementing permutation operations, or binding/unbinding via shifting (e.g. right/left, respectively).
/// # Arguments
/// * `vec` - The sparse binary vector to be shifted.
/// * `shift_by` - The number of positions to shift. Positive values shift to the right, negative values shift to the left.
/// # Returns
/// A new sparse binary vector that has been cyclically shifted.
pub fn cyclic_shift(vec: &CsVec<i8>, shift_by: isize) -> CsVec<i8> {
    let size = vec.dim() as isize;
    let mut new_indices: Vec<usize> = Vec::new();

    for (index, _) in vec.iter() {
        let mut new_index = index as isize + shift_by;
        if new_index < 0 {
            new_index += size;
        } else if new_index >= size {
            new_index -= size;
        }


        new_indices.push(new_index as usize);
    }

    from_indices(vec.dim(), &new_indices)
}


/// Computes the similarity between two sparse binary vectors.
/// Similarity is defined as 1 - (Hamming distance / dimension).
/// # Arguments
/// * `vec1` - The first sparse binary vector.
/// * `vec2` - The second sparse binary vector.
/// # Returns
/// The similarity as a f64 value between 0.0 and 1.0
pub fn similarity(vec1: &CsVec<i8>, vec2: &CsVec<i8>) -> f64 {
    assert_eq!(vec1.dim(), vec2.dim(), "Vectors must be of the same dimension to compute similarity.");

    let sim = hamming_distance(vec1, vec2) as f64 / vec1.dim() as f64;

    1f64 - sim
}

