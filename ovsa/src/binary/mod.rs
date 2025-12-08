
pub mod binary {
    use sprs::CsVec;
    use rand::seq::index::sample;
    use rand::rng;

    /// Generates a sparse random binary vector of given size with a specified number of active (1) entries.
    /// This could probably be optimized to use bit fields instead of u8 vectors.
    /// # Arguments
    /// * `dimension` - The size of the vector.
    /// * `n_active` - The number of active (1) entries in the vector.
    /// # Returns
    /// A sparse binary vector represented as `CsVec<i8>`.
    pub fn sparse_random(dimension: usize, n_active: usize) -> CsVec<i8> {

        let mut rng = rng();
        let indices = sample(&mut rng, dimension, n_active).into_vec();
        let data = vec![1i8; n_active];

        CsVec::new(dimension, indices, data)
    }


    /// Computes the consensus sum of a slice of sparse binary vectors.
    /// The consensus sum is determined by taking the majority value at each index across all vectors.
    /// # Arguments
    /// * `vectors` - A slice of sparse binary vectors represented as `CsVec<i8>`.
    /// # Returns
    /// A sparse binary vector representing the consensus sum.
    pub fn consensus_sum(vectors: &[CsVec<i8>]) -> CsVec<i8> {
        let size = vectors[0].dim();
        let mut result_data = vec![0i8; size];

        for vec in vectors {
            for (index, &value) in vec.iter() {
                result_data[index] += value * 2 - 1;
            }
        }

        let indices: Vec<usize> = result_data.iter()
            .enumerate()
            .filter_map(|(index, &value)| if value > 0 { Some(index) } else { None })
            .collect();

        let data: Vec<i8> = indices.iter().map(|&i| result_data[i]).collect();

        CsVec::new(size, indices, data)
    }


    /// Bundles multiple sparse binary vectors into a single vector using consensus sum.
    /// # Arguments
    /// * `vectors` - A slice of sparse binary vectors represented as `CsVec<i8>`.
    /// # Returns
    /// A sparse binary vector representing the bundled result.
    pub fn bundle(vectors: &[CsVec<i8>]) -> CsVec<i8> {
        consensus_sum(vectors)
    }


    /// Binds two sparse binary vectors using element-wise XOR operation.
    /// # Arguments
    /// * `vec1` - The first sparse binary vector.
    /// * `vec2` - The second sparse binary vector.
    /// # Returns
    /// A sparse binary vector representing the bound result.
    pub fn bind(vec1: &CsVec<i8>, vec2: &CsVec<i8>) -> CsVec<i8> {
        assert_eq!(vec1.dim(), vec2.dim(), "Vectors must be of the same dimension for binding.");

        let size = vec1.dim();
        let mut result = vec1 + vec2;
        let indices: Vec<usize> = result.iter()
            // XOR operation: 1 + 1 = 0, so we keep only entries with value 1
            .filter_map(|(index, &value)| if value == 1 { Some(index) } else { None })
            .collect();

        let data: Vec<i8> = indices.iter().map(|&i| 1i8).collect();

        CsVec::new(size, indices, data)
    }



}