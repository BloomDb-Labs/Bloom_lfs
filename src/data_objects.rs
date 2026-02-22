//! Filepath: src/data_objects.rs
//!
//! Delta and Base Page objects, LLAMA's primary data units
//!
//!
//! Access methods define the data withing the delta objects
//! to borrow a term from the flush buffer, we designate the
//! the data as a delta's [`payload`]. LLAMA knows nothing about
//! an objects payload.
//!
//! There are states however, that are visible LLAMA. LLAMA can
//! recognize whether a delta's type whther it may be a:
//!     
//!     - Flush Delta
//!     - Partial Swap Delta
//!     - Update Delta
//!
//! It can also retreive data relating to a Delta Record's
//!
//!     - Size
//!     - Page Id
//!     - Flush State
//!     - Payload Length
//!     - Pointers to prior and last flushed states
//! //!
//! ```text

//! - [`DeltaData`] owns the raw wire bytes and exposes field accessors.
//!   It is *type-blind* – it knows nothing about payload semantics.
//!
//! - [`Delta`] is the **object-safe trait** (analogous to Arrow's `Array`
//!   trait). Every concrete delta type implements it, providing:
//!     - `as_any`     → enables `downcast_ref` back to the concrete type.
//!     - `data`       → shared access to the underlying [`DeltaData`].
//!     - `delta_type` → discriminant without a downcast.
//!
//! - [`DeltaRef`] (`Arc<dyn Delta>`) is the owned, cheaply-cloned handle
//!   passed around the system – analogous to Arrow's `ArrayRef`.
//!
//! - Concrete structs (`FlushDelta`, `UpdateDelta`, `PartialSwapDelta`)
//!   hold nothing but an `Arc<DeltaData>`, making them zero-overhead
//!   wrappers that add type-safe accessor methods (e.g.
//!   `PartialSwapDelta::last_flushed()`).

use bytes::Bytes;
use std::{any::Any, ops::Deref, sync::Arc};

/// Base page default size of 4kb
pub const FOUR_KB_PAGE: usize = 4096;

///  8kb Base Page
pub const PAGE_SIZE_2: usize = FOUR_KB_PAGE * 2;

///  16kb Base Page
pub const PAGE_SIZE_4: usize = FOUR_KB_PAGE * 4;

///  64kb Base Page
pub const PAGE_SIZE_16: usize = FOUR_KB_PAGE * 16;

/// Llama Base Page
pub struct BasePage {
    state: [u8],
}

// ── Packed-bits constants ─────────────────────────────────────────────────────
//
//  u8 layout (stored at OFF_PACKED in the wire buffer):
//
//  ┌───┬───┬───┬───┬───┬─────────────────┐
//  │ F │ P │ U │ E │ R │   (reserved)    │
//  └───┴───┴───┴───┴───┴─────────────────┘
//    7   6   5   4   3     2   1   0

/// `F` – Flush-done: delta has been durably flushed.
pub const BIT_FLUSH_DONE: u8 = 1 << 7;

/// `P` – Partial-swap: page contents have been partially evicted.
pub const BIT_PARTIAL_SWAP: u8 = 1 << 6;

/// `U` – Update: delta carries an insert / update / delete.
pub const BIT_UPDATE: u8 = 1 << 5;

/// `E` – Failed-flush: flush attempt failed; retry required.
pub const BIT_FAILED_FLUSH: u8 = 1 << 4;

/// `R` – Reclaim: delta is eligible for consolidation.
pub const BIT_RECLAIM: u8 = 1 << 3;

//  Delta Format
//
//  ┌──────┬──────────┬──────┬──────────────┬─────────────────┬──────────────────┬──────┬─────────┐
//  │ Len  │ LSS Off  │ PID  │ Packed Bits  │ Prior State Ptr │ Last Flushed Ptr │ PLen │ Payload │
//  │ u32  │ u32      │ u32  │ u8           │ u64             │ u64              │ u32  │  [u8]   │
//  └──────┴──────────┴──────┴──────────────┴─────────────────┴──────────────────┴──────┴─────────┘
//     0       4         8        12               13                 21              29      33

const OFF_LEN: usize = 0;
const OFF_LSS: usize = 4;
const OFF_PID: usize = 8;
const OFF_PACKED: usize = 12;
const OFF_PRIOR_STATE: usize = 13;
const OFF_LAST_FLUSHED: usize = 21;
const OFF_PAYLOAD_LEN: usize = 29;
const OFF_PAYLOAD: usize = 33;

// I/O helpers

#[inline]
fn read_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}
#[inline]
fn read_u64(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(buf[off..off + 8].try_into().unwrap())
}
#[inline]
fn write_u32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}
#[inline]
fn write_u64(buf: &mut [u8], off: usize, v: u64) {
    buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// Assemble a complete wire buffer from its parts.
fn build_wire(
    lss_offset: u32,
    pid: u32,
    packed_bits: u8,
    prior_state: u64,
    last_flushed: u64,
    payload: &[u8],
) -> Bytes {
    let total = OFF_PAYLOAD + payload.len();
    let mut buf = vec![0u8; total];
    write_u32(&mut buf, OFF_LEN, total as u32);
    write_u32(&mut buf, OFF_LSS, lss_offset);
    write_u32(&mut buf, OFF_PID, pid);
    buf[OFF_PACKED] = packed_bits;
    write_u64(&mut buf, OFF_PRIOR_STATE, prior_state);
    write_u64(&mut buf, OFF_LAST_FLUSHED, last_flushed);
    write_u32(&mut buf, OFF_PAYLOAD_LEN, payload.len() as u32);
    buf[OFF_PAYLOAD..].copy_from_slice(payload);
    Bytes::from(buf)
}
/// Shared, type-blind wire-format buffer.
#[derive(Debug, Clone)]
pub struct DeltaData {
    ///
    ///  `Payload`
    ///  Operation-specific data.
    ///
    ///  Examples:
    ///  - Insert delta → key/value pair
    ///  - Delete delta → key tombstone
    ///  - Update delta → key + new value
    ///
    ///   In the case of B-Tree like access methods
    ///  - Split delta → split key + right sibling pointer
    ///  - Merge delta → merge boundary metadata
    ///
    ///  Payload layout depends on delta type and must be
    ///  interpretable without external state.
    ///
    ///  
    pub(crate) bytes: Bytes,
}

impl DeltaData {
    fn new(
        lss_offset: u32,
        pid: u32,
        packed_bits: u8,
        prior_state: u64,
        last_flushed: u64,
        payload: &[u8],
    ) -> Self {
        Self {
            bytes: build_wire(
                lss_offset,
                pid,
                packed_bits,
                prior_state,
                last_flushed,
                payload,
            ),
        }
    }

    /// Total byte length of this delta (header + payload).
    pub fn len(&self) -> usize {
        read_u32(&self.bytes, OFF_LEN) as usize
    }

    /// LSS offset – position of this record on secondary storage.
    pub fn lss_offset(&self) -> usize {
        read_u32(&self.bytes, OFF_LSS) as usize
    }

    /// Page identifier.
    pub fn pid(&self) -> u32 {
        read_u32(&self.bytes, OFF_PID)
    }

    /// Raw packed-bits byte. AND with the `BIT_*` constants to test flags.
    pub fn packed_bits(&self) -> u8 {
        self.bytes[OFF_PACKED]
    }

    /// Opaque pointer to the previous state in the version chain.
    pub fn prior_state_ptr(&self) -> usize {
        read_u64(&self.bytes, OFF_PRIOR_STATE) as usize
    }

    /// Pointer to the last stably-flushed state (set by `PartialSwapDelta`).
    pub fn last_flushed_ptr(&self) -> usize {
        read_u64(&self.bytes, OFF_LAST_FLUSHED) as usize
    }

    /// Immutable view of the payload bytes.
    pub fn payload(&self) -> &[u8] {
        let plen = read_u32(&self.bytes, OFF_PAYLOAD_LEN) as usize;
        &self.bytes[OFF_PAYLOAD..OFF_PAYLOAD + plen]
    }

    // Bit Flag accessors

    /// `F` – delta has been durably flushed.
    pub fn is_durable(&self) -> bool {
        self.bytes[OFF_PACKED] & BIT_FLUSH_DONE != 0
    }
    /// `P` – page was partially evicted when this delta was written.
    pub fn is_partial_swap(&self) -> bool {
        self.bytes[OFF_PACKED] & BIT_PARTIAL_SWAP != 0
    }
    /// `U` – delta carries an update operation.
    pub fn is_update(&self) -> bool {
        self.bytes[OFF_PACKED] & BIT_UPDATE != 0
    }
    /// `E` – the flush attempt for this delta failed.
    pub fn is_flush_failed(&self) -> bool {
        self.bytes[OFF_PACKED] & BIT_FAILED_FLUSH != 0
    }
    /// `R` – delta is eligible for consolidation / reclamation.
    pub fn is_reclaimable(&self) -> bool {
        self.bytes[OFF_PACKED] & BIT_RECLAIM != 0
    }
}

/// Discriminant returned by [`Delta::delta_type`].
///
/// Lets callers branch on delta kind without a downcast.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaKind {
    Flush = 0,
    PartialSwap = 1,
    Update = 2,
}

/// # Delta
///
/// /// Object-safe trait implemented by every concrete delta type.
///
/// A `Delta` represents an incremental modification to a page.
///
/// Deltas are **prepended** to a Base Page or to another Delta,
/// forming a singly-linked logical version chain:
///
/// ```text
/// Delta(N) -> Delta(N-1) -> ... -> Delta(0) -> Base Page
/// ```
///
/// Each delta describes a transformation from its `Prior State`.
/// The most recent state of a page is obtained by walking the chain
/// from head to base and applying deltas in order.
///
/// Access methods may append specialized delta payloads depending on the operation.
///
/// # Physical Layout
///
/// ```text
/// ┌─────────┬──────────────┬──────────────┬──────────────┬─────────────┬─────────┐
/// │ Len     │ LSN Offset   │ Prior State  │ Last Flushed │ Packed Bits │ Payload │
/// └─────────┴──────────────┴──────────────┴──────────────┴─────────────┴─────────┘
/// ```
///
/// ## Field Descriptions
///
/// ### `Len`
/// Total byte length of this delta (including header and payload).
/// Used for traversal, validation, and bounds checking.
///
/// ---
///
/// ### `LSN Offset`
/// Log Sequence Number associated with this delta.
/// Used for:
/// - Write-Ahead Logging (WAL) validation
/// - Crash recovery ordering
/// - Durability guarantees
///
/// The LSN must be persisted before this delta is considered durable.
///
/// ---
///
/// ### `Prior State`
/// Pointer (or page identifier) to the previous state in the chain.
/// This may reference:
///     - A Base Page
///     - Another Delta
/// either in memory or on secondary storsgr
///
///
/// Logically forms a version chain.
///
/// ---
///
/// ### `Last Flushed`
/// Indicates the highest LSN known to be flushed to stable storage
/// at the time this delta was written.
///
/// Used during:
/// - Recovery
/// - Flush coordination
/// - Cache management
///
/// ---
///
/// ### `u8 Packed Bits`
/// Bitfield containing state flags controlling delta semantics.
///
/// Layout:
///
/// ```text
/// [ F | P | U | E | R | The remaining bits are unused... ]
/// ```
///
/// Where:
///
/// - `F` → Flush-done bit  
///   - Set when the delta has been durably flushed.
///
/// - `P` → Partial-swap bit  
///   - Indicates that a pages contents have ben partially evicted
///
/// - `U` → Update bit  
///   - Indicates a change to a previous state
///     (e.g., insert/update/delete).
///
/// - `E` → Failed flush tag  
///   - Indicates the flush attempt failed and must be retried.
///
/// - `R` → Reclamation / Ready-for-compaction bit  
///   - Marks delta as eligible for consolidation.
///
/// Remaining bits are reserved for future extension.
///

pub trait Delta: Send + Sync {
    // --- type identity -------------------------------------------------------

    /// Returns `self` as `&dyn Any` so callers can `downcast_ref`.
    ///
    /// Every impl is identical: `fn as_any(&self) -> &dyn Any { self }`.
    fn as_any(&self) -> &dyn Any;

    /// Discriminant for branching without a downcast.
    fn delta_type(&self) -> u8;

    // --- storage access ------------------------------------------------------

    /// Returns a reference to the shared [`DeltaData`].
    ///
    /// All trait-default accessors below delegate here, so concrete types
    /// need only implement `as_any`, `delta_type`, and `data`.
    fn data(&self) -> &DeltaData;

    // --- forwarded accessors (default-implemented, no override needed) --------

    fn is_durable(&self) -> bool {
        self.data().is_durable()
    }
    fn lss_offset(&self) -> usize {
        self.data().lss_offset()
    }
    fn payload(&self) -> &[u8] {
        self.data().payload()
    }

    fn len(&self) -> usize;

    fn pid(&self) -> u32 {
        self.data().pid()
    }
    fn prior_state_ptr(&self) -> usize {
        self.data().prior_state_ptr()
    }
    fn packed_bits(&self) -> u8 {
        self.data().packed_bits()
    }
}

/// Owned, cheaply-cloned, type-erased delta handle.
///
/// Analogous to Arrow's `ArrayRef = Arc<dyn Array>`.
pub type DeltaRef = Arc<dyn Delta>;

// Each struct is a zero-overhead type around `Arc<DeltaData>`.
// They implement `Delta` via three trivial lines, then add any
// type-specific methods below the impl block.

/// Prepended when a page is flushed to stable storage (`F` bit).
pub struct FlushDelta {
    data: DeltaData,
}

impl FlushDelta {
    pub fn new(prior: &dyn Delta, pid: u32, lss_offset: u32) -> Self {
        Self {
            data: DeltaData::new(
                lss_offset,
                pid,
                BIT_FLUSH_DONE,
                prior.prior_state_ptr() as u64,
                0,
                &[],
            ),
        }
    }
}

impl Delta for FlushDelta {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn delta_type(&self) -> u8 {
        0
    }
    fn data(&self) -> &DeltaData {
        &self.data
    }
    fn len(&self) -> usize {
        self.data.len()
    }
}

/// Carries an insert / update / delete change to a prior state (`U` bit).
pub struct UpdateDelta {
    data: DeltaData,
}

impl UpdateDelta {
    pub fn new(payload: &[u8], prior_state: usize, pid: u32, lss_offset: u32) -> Self {
        Self {
            data: DeltaData::new(lss_offset, pid, BIT_UPDATE, prior_state as u64, 0, payload),
        }
    }
}

impl Delta for UpdateDelta {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn delta_type(&self) -> u8 {
        1
    }
    fn data(&self) -> &DeltaData {
        &self.data
    }
    fn len(&self) -> usize {
        self.data.len()
    }
}

/// Records that a page was partially evicted to secondary storage (`P` bit).
///
/// Carries two pointers: the prior in-memory / on-disk state, and the last
/// stably-flushed state. The second pointer enables recovery without
/// replaying the full chain, and is only accessible after downcasting to
/// this concrete type.
pub struct PartialSwapDelta {
    data: DeltaData,
}

impl PartialSwapDelta {
    pub fn new(
        payload: &[u8],
        prior_state: usize,
        last_flushed_state: usize,
        pid: u32,
        lss_offset: u32,
    ) -> Self {
        Self {
            data: DeltaData::new(
                lss_offset,
                pid,
                BIT_PARTIAL_SWAP,
                prior_state as u64,
                last_flushed_state as u64,
                payload,
            ),
        }
    }

    /// Pointer to the last stably-flushed state at the time of eviction.
    ///
    /// This method is intentionally absent from the `Delta` trait: it only
    /// makes sense on partial-swap deltas, so callers must downcast first.
    pub fn last_flushed(&self) -> usize {
        self.data.last_flushed_ptr()
    }
}

impl Delta for PartialSwapDelta {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn delta_type(&self) -> u8 {
        2
    }
    fn data(&self) -> &DeltaData {
        &self.data
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

impl Deref for UpdateDelta {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.data.bytes
    }
}

impl Deref for PartialSwapDelta {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.data.bytes
    }
}
impl Deref for FlushDelta {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.data.bytes
    }
}

// same for FlushDelta and PartialSwapDelta

// same for FlushDelta and PartialSwapDelta

// same for FlushDelta and PartialSwapDelta
/// Attempt to downcast a `DeltaRef` to `&T`.
///
/// Mirrors Arrow's `as_primitive_array` / `downcast_array` family.
///
/// ```rust
/// let d: DeltaRef = UpdateDelta::new(b"kv", 0, 1, 0);
/// let concrete = downcast_delta::<UpdateDelta>(&d).unwrap();
/// assert!(concrete.data().is_update());
/// ```
pub fn downcast_delta<T: Delta + 'static>(delta: &DeltaRef) -> Option<&T> {
    delta.as_any().downcast_ref::<T>()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flush_delta_sets_only_flush_bit() {
        let prior = UpdateDelta::new(b"prior", 0xDEAD, 1, 10);
        let d = FlushDelta::new(&prior, 1, 42);

        assert_eq!(d.delta_type(), DeltaKind::Flush as u8);
        assert!(d.is_durable());
        assert!(!d.data().is_partial_swap());
        assert!(!d.data().is_update());
        assert_eq!(d.packed_bits() & BIT_FLUSH_DONE, BIT_FLUSH_DONE);
    }

    #[test]
    fn update_delta_roundtrip() {
        let payload = b"hello world";
        let d = UpdateDelta::new(payload, 0x1234, 7, 99);

        assert_eq!(d.delta_type(), DeltaKind::Update as u8);
        assert!(d.data().is_update());
        assert!(!d.is_durable());
        assert_eq!(d.payload(), payload);
        assert_eq!(d.pid(), 7);
        assert_eq!(d.lss_offset(), 99);
        assert_eq!(d.prior_state_ptr(), 0x1234);
    }

    #[test]
    fn partial_swap_delta_roundtrip() {
        let d = PartialSwapDelta::new(b"swap", 0xAAAA, 0xBBBB, 3, 55);

        assert_eq!(d.delta_type(), DeltaKind::PartialSwap as u8);
        assert!(d.data().is_partial_swap());
        assert_eq!(d.payload(), b"swap");
        assert_eq!(d.prior_state_ptr(), 0xAAAA);
        assert_eq!(d.last_flushed(), 0xBBBB); // type-specific method
    }

    #[test]
    fn type_erased_dispatch_and_downcast() {
        let deltas: Vec<DeltaRef> = vec![
            Arc::new(UpdateDelta::new(b"upd", 1, 10, 0)),
            Arc::new(PartialSwapDelta::new(b"swp", 2, 3, 20, 1)),
            Arc::new(FlushDelta::new(&UpdateDelta::new(b"p", 0, 5, 0), 5, 2)),
        ];

        for d in &deltas {
            match d.delta_type() {
                0 => {
                    let c = downcast_delta::<UpdateDelta>(d).unwrap();
                    assert!(c.data().is_update());
                }
                1 => {
                    let c = downcast_delta::<PartialSwapDelta>(d).unwrap();
                    let _ = c.last_flushed(); // only reachable after downcast
                }
                2 => {
                    let c = downcast_delta::<FlushDelta>(d).unwrap();
                    assert!(c.is_durable());
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn wrong_downcast_returns_none() {
        let d: DeltaRef = Arc::new(UpdateDelta::new(b"x", 0, 0, 0));
        assert!(downcast_delta::<FlushDelta>(&d).is_none());
        assert!(downcast_delta::<PartialSwapDelta>(&d).is_none());
    }

    #[test]
    fn packed_bits_do_not_bleed() {
        let u = UpdateDelta::new(b"x", 0, 0, 0);
        assert_eq!(u.packed_bits() & BIT_FLUSH_DONE, 0, "update must not set F");
        assert_eq!(
            u.packed_bits() & BIT_PARTIAL_SWAP,
            0,
            "update must not set P"
        );

        let p = PartialSwapDelta::new(b"x", 0, 0, 0, 0);
        assert_eq!(
            p.packed_bits() & BIT_FLUSH_DONE,
            0,
            "partial-swap must not set F"
        );
        assert_eq!(
            p.packed_bits() & BIT_UPDATE,
            0,
            "partial-swap must not set U"
        );
    }
}
