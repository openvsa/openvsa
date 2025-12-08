
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

        let mut rng  = rng();
        let indices: Vec<usize> = sample(&mut rng, dimension, n_active).into_vec();
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

        CsVec::new(dimension, indices.to_vec(), data)
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

        let bound_vec = bind(vec1, vec2);
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
        let mut result_data: Vec<i8> = vec![0i8; size];

        for vec in vectors {
            for (index, &value) in vec.iter() {
                result_data[index] += value * 2 - 1;
            }
        }

        let indices: Vec<usize> = result_data.iter()
            .enumerate()
            .filter_map(|(index, &value)| if value > 0 { Some(index) } else { None })
            .collect();

        from_indices(size, &indices)
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

        let size: usize = vec1.dim();
        // to simulate an XOR operation, we add the two vectors and keep only the entries where the sum is 1
        let result: CsVec<i8> = vec1 + vec2;
        let indices: Vec<usize> = result.iter()
            // XOR operation: 1 + 1 = 0, so we keep only entries with value 1
            .filter_map(|(index, &value)| if value == 1 { Some(index) } else { None })
            .collect();

        from_indices(size, &indices)
    }


    pub fn unbind(vec1: &CsVec<i8>, vec2: &CsVec<i8>) -> CsVec<i8> {
        // In binary vectors, binding and unbinding are the same operation (XOR)
        bind(vec1, vec2)
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





}