use rand::seq::index::sample;
use rand::distr::Uniform;
use rand::rng;

use ovsa::binary::sparse_random;


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
    let dimension = 10;
    let vec1 = ovsa::binary::from_indices(dimension, &[1, 3, 5]);
    let vec2 = ovsa::binary::from_indices(dimension, &[3, 4, 5]);
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

#[test]
fn test_consensus_sum_tie() {
    let mut rng  = rng();
    let dimension = 100;

    let mut indices1: Vec<usize> = sample(&mut rng, dimension/2, dimension/4).into_vec();
    indices1.sort();

    let vec1 = ovsa::binary::from_indices(dimension, &indices1);

    let mut indices2: Vec<usize> = sample(&mut rng, dimension/2, dimension/4).into_vec();
    for element in &mut indices2{
        // offset by half the dimension to avoid overlap
        *element += dimension / 2;
    }
    indices2.sort();

    let vec2 = ovsa::binary::from_indices(dimension, &indices2);


    let consensus = ovsa::binary::consensus_sum(&[vec1, vec2]);
    let indices = consensus.indices();
    println!("Consensus indices: {:?}", indices);
    for &index in indices {
         // these are assigned randomly due to tie, how do we test?
        // assert_eq!(consensus[index], 0); // all positions should be 0 due to tie
        println!("Index {}: {}", index, consensus[index]);
    }
}