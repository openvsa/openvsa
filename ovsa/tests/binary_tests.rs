use rand::seq::index::sample;
use rand::rng;


#[test]
fn test_sparse_random() {
    let dimension = 10;
    let n_active = 3;
    let vec = ovsa::binary::sparse_random(dimension, n_active).expect("Failed to create sparse random vector");
    assert_eq!(vec.dim(), dimension);
    assert_eq!(vec.nnz(), n_active);
}

#[test]
fn test_from_indices() {
    let dimension = 10;
    let indices = vec![1, 3, 5];
    let vec = ovsa::binary::from_indices(dimension, &indices).expect("Failed to create vector from indices");
    assert_eq!(vec.dim(), dimension);
    assert_eq!(vec.nnz(), indices.len());
    for &index in &indices {
        assert_eq!(vec[index], 1);
    }
}

#[test]
fn test_hamming_distance() {
    let dimension = 10;
    let vec1 = ovsa::binary::from_indices(dimension, &[1, 3, 5]).unwrap();
    let vec2 = ovsa::binary::from_indices(dimension, &[3, 4, 5]).unwrap();
    let distance = ovsa::binary::hamming_distance(&vec1, &vec2);
    assert_eq!(distance, 2); // indices 1 and 4 are different
}

#[test]
fn test_consensus_sum() {
    let dimension = 10;
    let vec1 = ovsa::binary::from_indices(dimension, &[1, 3, 5]).unwrap();
    let vec2 = ovsa::binary::from_indices(dimension, &[3, 4, 5]).unwrap();
    let vec3 = ovsa::binary::from_indices(dimension, &[1, 6, 9]).unwrap();
    let consensus = ovsa::binary::consensus_sum(&[vec1, vec2, vec3]).expect("failed to compute consensus sum");
    let on_indices: Vec<usize> = vec![1, 3, 5];
    for index in 0..dimension {
        if on_indices.contains(&index) {
            assert_eq!(consensus[index], 1);
        }
    }
}

#[test]
fn test_consensus_sum_tie() {
    let mut rng  = rng();
    let dimension = 10000;

    let mut indices1: Vec<usize> = sample(&mut rng, dimension/2, dimension/4).into_vec();
    indices1.sort();

    let vec1 = ovsa::binary::from_indices(dimension, &indices1).unwrap();

    let mut indices2: Vec<usize> = sample(&mut rng, dimension/2, dimension/4).into_vec();
    for element in &mut indices2{
        // offset by half the dimension to avoid overlap
        *element += dimension / 2;
    }
    indices2.sort();

    let vec2 = ovsa::binary::from_indices(dimension, &indices2).unwrap();

    let consensus = ovsa::binary::consensus_sum(&[vec1, vec2]).unwrap();
    let indices = consensus.indices();

    println!("Consensus indices: n({}) should be close to {}", indices.len(), dimension / 4);

    let abs_diff = if indices.len() > dimension / 4 {
        indices.len() - dimension / 4
    } else {
        dimension / 4 - indices.len()
    };

    assert!(abs_diff < dimension / 20, "Consensus sum deviates too much from expected in tie case.");

}

#[test]
fn test_xor() {
    let dimension = 10;
    let vec1 = ovsa::binary::from_indices(dimension, &[1, 3, 5]).unwrap();
    let vec2 = ovsa::binary::from_indices(dimension, &[3, 4, 5]).unwrap();
    let result = ovsa::binary::xor(&vec1, &vec2).expect("failed to compute xor");
    let expected_indices = vec![1, 4];
    assert_eq!(result.nnz(), expected_indices.len());
    for &index in &expected_indices {
        assert_eq!(result[index], 1);
    }
}

#[test]
fn test_cyclic_shift() {
    let dimension = 10;
    let vec = ovsa::binary::from_indices(dimension, &[1, 3, 5, 9]).unwrap();
    let shifted_vec = ovsa::binary::cyclic_shift(&vec, 2);
    let expected_indices = vec![3, 5, 7, 1];
    assert_eq!(shifted_vec.nnz(), expected_indices.len());
    for &index in &expected_indices {
        assert_eq!(shifted_vec[index], 1);
    }
}

#[test]
fn test_cyclic_shift_negative() {
    let dimension = 10;
    let vec = ovsa::binary::from_indices(dimension, &[1, 3, 5, 0]).unwrap();
    let shifted_vec = ovsa::binary::cyclic_shift(&vec, -2);
    let expected_indices = vec![9, 1, 3, 8];
    assert_eq!(shifted_vec.nnz(), expected_indices.len());
    for &index in &expected_indices {
        assert_eq!(shifted_vec[index], 1);
    }
}

#[test]
fn test_similarity() {
    let dimension = 10;
    let vec1 = ovsa::binary::from_indices(dimension, &[1, 3, 5]).unwrap();
    let vec2 = ovsa::binary::from_indices(dimension, &[3, 4, 5]).unwrap();
    let similarity = ovsa::binary::similarity(&vec1, &vec2).expect("Failed to compute similarity");
    // two items are not the same (indices 1 and 4) out of four active indices total
    let expected_similarity = 1.0 - (2.0 / (dimension as f64));
    assert_eq!(similarity, expected_similarity); // 2 common active out of 4 total active
}

