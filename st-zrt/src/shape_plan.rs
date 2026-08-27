//! Finite, canonical serving-shape admission plan.
//!
//! A plan is built and sealed before serving. Exact classification hashes borrowed shape slices
//! without allocating, then verifies the complete shapes so hash collisions cannot misroute work.
//! CUDA graph users should normally keep the default [`FallbackPolicy::Strict`] policy: no live
//! request can create or capture a surprise graph.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Opaque index of a canonical bucket within a [`ServingShapePlan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShapeId(u32);

impl ShapeId {
    /// Convert this id to an index for plan/lane lookup.
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    pub(crate) fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("zrt: serving shape count exceeds u32::MAX"))
    }
}

/// Deliberate output placement for one canonical bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputPolicy {
    /// Reusable ordinary host buffers.
    #[default]
    HostBuffer,
    /// CUDA-pinned host buffers for asynchronous transfers.
    CudaPinned,
    /// Reusable device-resident outputs.
    DeviceResident,
}

/// One finite canonical bucket in a serving plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalShape {
    id: ShapeId,
    input_shapes: Vec<Vec<i64>>,
    output_shapes: Vec<Vec<i64>>,
    output_policy: OutputPolicy,
}

impl CanonicalShape {
    #[inline]
    pub const fn id(&self) -> ShapeId {
        self.id
    }

    #[inline]
    pub fn input_shapes(&self) -> &[Vec<i64>] {
        &self.input_shapes
    }

    #[inline]
    pub fn output_shapes(&self) -> &[Vec<i64>] {
        &self.output_shapes
    }

    #[inline]
    pub const fn output_policy(&self) -> OutputPolicy {
        self.output_policy
    }
}

/// Policy for a request that is not an exact canonical shape.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FallbackPolicy {
    /// Reject it. This is the safe default for sealed CUDA graph serving.
    #[default]
    Strict,
    /// Select the smallest component-wise fitting bucket, provided its padding overhead does not
    /// exceed `max_padding_waste_ratio` (`0.25` means at most 25% extra input elements).
    PadToNearest { max_padding_waste_ratio: f32 },
    /// Signal that the caller should use a separate non-graph fallback session.
    FallbackSession,
}

/// Classification failure. Error paths may allocate diagnostic shape copies; successful exact
/// classification does not allocate.
#[derive(Debug, Clone, PartialEq)]
pub enum ClassifyError {
    NotInPlan,
    NoFittingShape,
    PaddingWasteExceeded {
        actual_ratio: f64,
        maximum_ratio: f32,
    },
}

impl std::fmt::Display for ClassifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotInPlan => f.write_str("shape is not in the sealed serving plan"),
            Self::NoFittingShape => f.write_str("no canonical serving shape fits the request"),
            Self::PaddingWasteExceeded {
                actual_ratio,
                maximum_ratio,
            } => write!(
                f,
                "nearest serving shape padding waste {actual_ratio:.3} exceeds limit {maximum_ratio:.3}"
            ),
        }
    }
}

impl std::error::Error for ClassifyError {}

/// Error while constructing an invalid serving plan.
#[derive(Debug, Clone, PartialEq)]
pub enum ShapePlanError {
    EmptyPlan,
    EmptyInputSet,
    NonPositiveDimension { kind: &'static str, dimension: i64 },
    DuplicateShape,
    TooManyShapes,
    InvalidPaddingWasteRatio(f32),
}

impl std::fmt::Display for ShapePlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPlan => f.write_str("serving shape plan must contain at least one bucket"),
            Self::EmptyInputSet => f.write_str("canonical bucket must contain at least one input"),
            Self::NonPositiveDimension { kind, dimension } => {
                write!(
                    f,
                    "canonical {kind} dimension must be positive, got {dimension}"
                )
            },
            Self::DuplicateShape => f.write_str("duplicate canonical input shape"),
            Self::TooManyShapes => f.write_str("serving shape plan exceeds u32::MAX buckets"),
            Self::InvalidPaddingWasteRatio(ratio) => write!(
                f,
                "max_padding_waste_ratio must be finite and non-negative, got {ratio}"
            ),
        }
    }
}

impl std::error::Error for ShapePlanError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ShapeHash(u64);

impl ShapeHash {
    #[inline]
    fn from_slices(shapes: &[&[i64]]) -> Self {
        const OFFSET: u64 = 0xcbf29ce484222325;
        const PRIME: u64 = 0x100000001b3;
        let mut hash = OFFSET;
        hash = (hash ^ shapes.len() as u64).wrapping_mul(PRIME);
        for shape in shapes {
            hash = (hash ^ shape.len() as u64).wrapping_mul(PRIME);
            for &dim in *shape {
                hash = (hash ^ dim as u64).wrapping_mul(PRIME);
            }
        }
        Self(hash)
    }
}

#[derive(Default)]
struct ShapeHashHasher(u64);

impl Hasher for ShapeHashHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Fallback for Hash implementations that do not use write_u64.
        const PRIME: u64 = 0x100000001b3;
        let mut hash = 0xcbf29ce484222325;
        for &byte in bytes {
            hash = (hash ^ u64::from(byte)).wrapping_mul(PRIME);
        }
        self.0 = hash;
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

type ShapeIndex = HashMap<ShapeHash, Vec<ShapeId>, BuildHasherDefault<ShapeHashHasher>>;

/// Immutable, finite canonical shape set shared by schedulers and serving lanes.
#[derive(Debug, Clone)]
pub struct ServingShapePlan {
    buckets: Vec<CanonicalShape>,
    // A hash may map to multiple ids. Full shape equality below makes collisions harmless.
    index: ShapeIndex,
    fallback: FallbackPolicy,
}

impl ServingShapePlan {
    pub fn builder() -> ServingShapePlanBuilder {
        ServingShapePlanBuilder::default()
    }

    /// Classify borrowed input shape slices without allocating on an exact-match success path.
    #[inline]
    pub fn classify(&self, input_shapes: &[&[i64]]) -> Result<ShapeId, ClassifyError> {
        let hash = ShapeHash::from_slices(input_shapes);
        if let Some(candidates) = self.index.get(&hash) {
            for &id in candidates {
                if shapes_equal(&self.buckets[id.as_usize()].input_shapes, input_shapes) {
                    return Ok(id);
                }
            }
        }

        match self.fallback {
            FallbackPolicy::Strict | FallbackPolicy::FallbackSession => {
                Err(ClassifyError::NotInPlan)
            },
            FallbackPolicy::PadToNearest {
                max_padding_waste_ratio,
            } => self.classify_padded(input_shapes, max_padding_waste_ratio),
        }
    }

    fn classify_padded(
        &self, requested: &[&[i64]], maximum_ratio: f32,
    ) -> Result<ShapeId, ClassifyError> {
        let mut best: Option<(u128, u128, ShapeId)> = None;
        for bucket in &self.buckets {
            let Some((requested_elements, canonical_elements)) =
                fitting_element_counts(requested, &bucket.input_shapes)
            else {
                continue;
            };
            let waste = canonical_elements.saturating_sub(requested_elements);
            if best.is_none_or(|(best_waste, best_total, _)| {
                (waste, canonical_elements) < (best_waste, best_total)
            }) {
                best = Some((waste, canonical_elements, bucket.id));
            }
        }
        let Some((waste, _, id)) = best else {
            return Err(ClassifyError::NoFittingShape);
        };
        let requested_elements =
            requested_element_count(requested).ok_or(ClassifyError::NoFittingShape)?;
        let actual_ratio = waste as f64 / requested_elements as f64;
        if actual_ratio > f64::from(maximum_ratio) {
            return Err(ClassifyError::PaddingWasteExceeded {
                actual_ratio,
                maximum_ratio,
            });
        }
        Ok(id)
    }

    #[inline]
    pub fn bucket(&self, id: ShapeId) -> Option<&CanonicalShape> {
        self.buckets.get(id.as_usize())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.buckets.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    #[inline]
    pub const fn fallback_policy(&self) -> FallbackPolicy {
        self.fallback
    }

    pub fn buckets(&self) -> impl ExactSizeIterator<Item = &CanonicalShape> {
        self.buckets.iter()
    }
}

/// Startup-only builder for [`ServingShapePlan`].
#[derive(Debug, Default)]
pub struct ServingShapePlanBuilder {
    buckets: Vec<(Vec<Vec<i64>>, Vec<Vec<i64>>, OutputPolicy)>,
    fallback: FallbackPolicy,
}

impl ServingShapePlanBuilder {
    pub fn fallback_policy(&mut self, policy: FallbackPolicy) -> &mut Self {
        self.fallback = policy;
        self
    }

    pub fn add_shape(
        &mut self, input_shapes: impl IntoIterator<Item = Vec<i64>>,
        output_shapes: impl IntoIterator<Item = Vec<i64>>, output_policy: OutputPolicy,
    ) -> &mut Self {
        self.buckets.push((
            input_shapes.into_iter().collect(),
            output_shapes.into_iter().collect(),
            output_policy,
        ));
        self
    }

    pub fn build(self) -> Result<ServingShapePlan, ShapePlanError> {
        if self.buckets.is_empty() {
            return Err(ShapePlanError::EmptyPlan);
        }
        if self.buckets.len() > u32::MAX as usize {
            return Err(ShapePlanError::TooManyShapes);
        }
        if let FallbackPolicy::PadToNearest {
            max_padding_waste_ratio,
        } = self.fallback
        {
            if !max_padding_waste_ratio.is_finite() || max_padding_waste_ratio < 0.0 {
                return Err(ShapePlanError::InvalidPaddingWasteRatio(
                    max_padding_waste_ratio,
                ));
            }
        }

        let mut buckets: Vec<CanonicalShape> = Vec::with_capacity(self.buckets.len());
        let mut index =
            ShapeIndex::with_capacity_and_hasher(self.buckets.len(), BuildHasherDefault::default());
        for (position, (input_shapes, output_shapes, output_policy)) in
            self.buckets.into_iter().enumerate()
        {
            validate_shapes(&input_shapes, &output_shapes)?;
            let borrowed: Vec<&[i64]> = input_shapes.iter().map(Vec::as_slice).collect();
            let hash = ShapeHash::from_slices(&borrowed);
            if let Some(ids) = index.get(&hash) {
                if ids
                    .iter()
                    .any(|id| buckets[id.as_usize()].input_shapes == input_shapes)
                {
                    return Err(ShapePlanError::DuplicateShape);
                }
            }
            let id = ShapeId::from_index(position);
            buckets.push(CanonicalShape {
                id,
                input_shapes,
                output_shapes,
                output_policy,
            });
            index.entry(hash).or_default().push(id);
        }
        Ok(ServingShapePlan {
            buckets,
            index,
            fallback: self.fallback,
        })
    }
}

fn validate_shapes(inputs: &[Vec<i64>], outputs: &[Vec<i64>]) -> Result<(), ShapePlanError> {
    if inputs.is_empty() {
        return Err(ShapePlanError::EmptyInputSet);
    }
    for &dimension in inputs.iter().flatten() {
        if dimension <= 0 {
            return Err(ShapePlanError::NonPositiveDimension {
                kind: "input",
                dimension,
            });
        }
    }
    for &dimension in outputs.iter().flatten() {
        if dimension <= 0 {
            return Err(ShapePlanError::NonPositiveDimension {
                kind: "output",
                dimension,
            });
        }
    }
    Ok(())
}

#[inline]
fn shapes_equal(canonical: &[Vec<i64>], requested: &[&[i64]]) -> bool {
    canonical.len() == requested.len()
        && canonical
            .iter()
            .zip(requested)
            .all(|(left, right)| left.as_slice() == *right)
}

fn requested_element_count(shapes: &[&[i64]]) -> Option<u128> {
    let mut sum = 0_u128;
    for shape in shapes {
        let mut product = 1_u128;
        for &dimension in *shape {
            let positive = u128::try_from(dimension).ok().filter(|&d| d != 0)?;
            product = product.checked_mul(positive)?;
        }
        sum = sum.checked_add(product)?;
    }
    (sum != 0).then_some(sum)
}

fn fitting_element_counts(requested: &[&[i64]], canonical: &[Vec<i64>]) -> Option<(u128, u128)> {
    if requested.len() != canonical.len() {
        return None;
    }
    let mut requested_sum = 0_u128;
    let mut canonical_sum = 0_u128;
    for (request, bucket) in requested.iter().zip(canonical) {
        if request.len() != bucket.len() {
            return None;
        }
        let mut request_product = 1_u128;
        let mut bucket_product = 1_u128;
        for (&actual, &planned) in request.iter().zip(bucket) {
            if actual <= 0 || planned < actual {
                return None;
            }
            request_product = request_product.checked_mul(actual as u128)?;
            bucket_product = bucket_product.checked_mul(planned as u128)?;
        }
        requested_sum = requested_sum.checked_add(request_product)?;
        canonical_sum = canonical_sum.checked_add(bucket_product)?;
    }
    Some((requested_sum, canonical_sum))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(policy: FallbackPolicy) -> ServingShapePlan {
        let mut builder = ServingShapePlan::builder();
        builder.fallback_policy(policy);
        builder
            .add_shape(
                [vec![1, 8], vec![1, 8]],
                [vec![1, 384]],
                OutputPolicy::HostBuffer,
            )
            .add_shape(
                [vec![32, 64], vec![32, 64]],
                [vec![32, 384]],
                OutputPolicy::DeviceResident,
            )
            .add_shape(
                [vec![64, 128], vec![64, 128]],
                [vec![64, 384]],
                OutputPolicy::CudaPinned,
            );
        builder.build().unwrap()
    }

    #[test]
    fn exact_classification_is_stable() {
        let plan = plan(FallbackPolicy::Strict);
        assert_eq!(plan.classify(&[&[32, 64], &[32, 64]]), Ok(ShapeId(1)));
        assert_eq!(
            plan.bucket(ShapeId(1)).unwrap().output_policy(),
            OutputPolicy::DeviceResident
        );
    }

    #[test]
    fn strict_and_fallback_session_signal_a_miss() {
        for policy in [FallbackPolicy::Strict, FallbackPolicy::FallbackSession] {
            assert_eq!(
                plan(policy).classify(&[&[16, 32], &[16, 32]]),
                Err(ClassifyError::NotInPlan)
            );
        }
    }

    #[test]
    fn padding_selects_smallest_fitting_bucket() {
        let plan = plan(FallbackPolicy::PadToNearest {
            max_padding_waste_ratio: 3.0,
        });
        assert_eq!(plan.classify(&[&[16, 32], &[16, 32]]), Ok(ShapeId(1)));
    }

    #[test]
    fn padding_limit_is_enforced() {
        let plan = plan(FallbackPolicy::PadToNearest {
            max_padding_waste_ratio: 0.25,
        });
        assert!(matches!(
            plan.classify(&[&[16, 32], &[16, 32]]),
            Err(ClassifyError::PaddingWasteExceeded { .. })
        ));
    }

    #[test]
    fn builder_rejects_duplicate_and_dynamic_shapes() {
        let mut duplicate = ServingShapePlan::builder();
        duplicate
            .add_shape([vec![1, 8]], [vec![1, 3]], OutputPolicy::HostBuffer)
            .add_shape([vec![1, 8]], [vec![1, 9]], OutputPolicy::HostBuffer);
        assert_eq!(
            duplicate.build().unwrap_err(),
            ShapePlanError::DuplicateShape
        );

        let mut dynamic = ServingShapePlan::builder();
        dynamic.add_shape([vec![-1, 8]], [vec![1, 3]], OutputPolicy::HostBuffer);
        assert!(matches!(
            dynamic.build(),
            Err(ShapePlanError::NonPositiveDimension { .. })
        ));
    }
}
