


use ovsa::binary::sparse_random;

// mod binary;

#[test]
fn test_sparse_random() {
    let dimension = 10;
    let n_active = 3;
    let vec = sparse_random(dimension, n_active);
    assert_eq!(vec.dim(), dimension);
    assert_eq!(vec.nnz(), n_active);
}

#[test]
fn test_from_indices() {
    let dimension = 10;
    let indices = vec![1, 3, 5];
    let vec = ovsa::binary::from_indices(dimension, &indices);
    assert_eq!(vec.dim(), dimension);
    assert_eq!(vec.nnz(), indices.len());
    for &index in &indices {
        assert_eq!(vec[index], 1);
    }
}

#[test]
fn test_hamming_distance() {
    let vec1 = ovsa::binary::from_indices(10, &[1, 3, 5]);
    let vec2 = ovsa::binary::from_indices(10, &[3, 4, 5]);
    let distance = ovsa::binary::hamming_distance(&vec1, &vec2);
    assert_eq!(distance, 2); // indices 1 and 4 are different
}

#[test]
fn test_consensus_sum() {
    let dimension = 10;
    let vec1 = ovsa::binary::from_indices(dimension, &[1, 3, 5]);
    let vec2 = ovsa::binary::from_indices(dimension, &[3, 4, 5]);
    let vec3 = ovsa::binary::from_indices(dimension, &[1, 6, 9]);
    let consensus = ovsa::binary::consensus_sum(&[vec1, vec2, vec3]);
    let on_indices: Vec<usize> = vec![1, 3, 5];
    for index in 0..dimension {
        if on_indices.contains(&index) {
            assert_eq!(consensus[index], 1);
        }
    }
}