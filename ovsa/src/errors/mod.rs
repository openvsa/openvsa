#[derive(Debug, Clone)]

pub enum OVSAError {
    VectorSizeMismatch,
    EmptyVectorList,
    EmptyIndices,
    ZeroActiveElements,
    ZeroDimension,
    TooManyActiveElements,
}
