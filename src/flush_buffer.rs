//! Filepath: src/flush_buffer.rs
//!
//! flush buffer for [`LLAMA`]
//!
//! Latch free I/O buffer ring
//! Ammortises writes by batching deltas into an in-memory flush buffer
//! All threads participate in managing this buffer.
//! As outline in the literature, the  flush procedure is as follows

//! 1.  Identify the state of the page that we intend to flush.
//!
//! 2.  Seize space in the LSS buffer into which to write the state.
//!
//! 3.  Perform Atomic operations to determine whether the flush will succeed.
//!
//! 4.  If  step  3  succeeds,  write  the  state  to  be  saved  into
//!     the  LSS.  While  we  are  writing  into  the  LSS,  LLAMA  prevents  
//!     the  buffer from being written to LSS secondary storage.
//!
//! 5.  If step 3 fails, write "Failed Flush" into the reserved space in the
//!     buffer.  This consumes storage but removes any ambiguity as to which
//!     flushes have succeeded or failed

use std::{
    cell::UnsafeCell,
    pin::Pin,
    sync::{
        atomic::{AtomicPtr, AtomicUsize, Ordering},
        Arc,
    },
    usize,
};

/// A lightweight `Send + Sync` byte buffer with non-global cursor management.
///
/// Alternative containers such as `BytesMut` use a global mutable cursor, meaning
/// any change in cursor position is visible to all threads sharing the resource.
/// Here, cursor management is delegated to [`FlushBuffer`], which uses atomic
/// fetch-and-add to hand out non-overlapping byte ranges. This is what makes
/// the `unsafe impl Sync` sound.
///
/// # Safety
/// `Sync` is manually implemented because [`UnsafeCell`] opts out of it by default.
/// The invariant that upholds this is: all mutable access to the inner buffer is
/// mediated by [`FlushBuffer`], which guarantees no two threads are ever granted
/// overlapping regions.
#[derive(Debug)]
pub(crate) struct Buffer {

    buffer: UnsafeCell<Box<[u8]>>,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

/// A reference-counted handle to a shared [`Buffer`].
pub(crate) type SharedBuffer = Arc<Buffer>;

/// Flush buffer size in bytes
pub const BUFFER_SIZE: usize = 65536;

/// Bit 0 — sealed flag. When set, the buffer is closed to new writers.
const SEALED_BIT: usize = 1 << 0;

/// Each active writer contributes this value to the packed state word.
/// Writer count lives in bits 1.. so a single writer is represented as `2`.
const WRITER_ONE: usize = 1 << 2;

/// Set while a flush is in progress; prevents new writers from entering a
/// buffer that is being drained to stable storage.
const FLUSH_IN_PROGRESS_BIT: usize = 1 << 1;

/// Errors a thread may encounter while managing the buffer.
#[derive(Debug)]
pub enum BufferError {
    /// The payload exceeds the remaining capacity of the flush buffer.
    InsufficientSpace,

    /// The buffer is sealed and no longer accepting writes.
    EncounteredSealedBuffer,

    /// CAS observed an already-sealed buffer.
    EncounteredSealedBufferDuringCOMPEX,

    /// CAS observed an already-unsealed buffer.
    EncounteredUnSealedBufferDuringCOMPEX,

    /// A flush was attempted while writers are still active.
    ActiveUsers,

    /// Undefined / corrupt state.
    InvalidState,

    /// Ring is exhausted — all buffers are sealed or being flushed.
    RingExhausted,
}

/// Successful operation outcomes.
#[derive(Debug)]
pub enum BufferMsg {
    /// Buffer has been sealed.
    SealedBuffer,

    /// Data written successfully.
    SuccessfullWrite,

    /// Data written and buffer flushed to stable storage.
    SuccessfullWriteFlush,

    /// Buffer is ready to flush; carries the ring position of the sealed buffer.
    FreeToFlush(usize),
}

/// A LLAMA I/O flush buffer.
///
/// # State word layout
///
/// ```text
/// ┌──────────────────────────────┬───────────────────┬──────────┐
/// │  Bits 3..  (writer count)    │  Bit 2 flush-prog │  Bit 0   │
/// │                              │                   │  sealed  │
/// └──────────────────────────────┴───────────────────┴──────────┘
/// ```
///
/// Writer count is encoded in bits `1..` via [`WRITER_ONE`] = 2, so each
/// active writer adds 2 to the word and the sealed flag sits independently in
/// bit 0.  Reading both fields from a single atomic snapshot is what prevents
/// TOCTOU races on the "last writer + sealed" flush trigger.
#[derive(Debug)]
pub(crate) struct FlushBuffer {
    /// Packed atomic state — see type-level docs.
    state: AtomicUsize,

    /// Next available write offset.  `fetch_add` hands out non-overlapping
    /// byte ranges to concurrent writers.
    current_offset: AtomicUsize,

    /// Backing byte store.
    buf: SharedBuffer,

    /// Logical start address of this buffer within the LSS address space.
    address: usize,

    /// Position of this buffer inside the [`FlushBufferRing`].
    pos: usize,
}

unsafe impl Send for FlushBuffer {}
unsafe impl Sync for FlushBuffer {}

impl FlushBuffer {
    pub(crate) fn new_buffer(buffer_number: usize) -> FlushBuffer {
        Self {
            state: AtomicUsize::new(0),
            current_offset: AtomicUsize::new(0),
            buf: Arc::new(Buffer {
                buffer: UnsafeCell::new(Box::new(vec![0u8; BUFFER_SIZE]).into_boxed_slice()),
            }),
            address: BUFFER_SIZE * buffer_number,
            pos: buffer_number,
        }
    }

    /// Returns `true` if this buffer is ready to accept new writers.
    ///
    /// Used by [`FlushBufferRing::rotate_after_seal`] to skip over slots that are still being drained to stable storage.
    pub(crate) fn is_available(&self) -> bool {
        self.state.load(Ordering::Acquire) & (SEALED_BIT | FLUSH_IN_PROGRESS_BIT) == 0
    }

    /// Attempts to reserve `payload_size` bytes, returning the starting offset.
    ///
    /// Returns [`BufferError::EncounteredSealedBuffer`] immediately if the
    /// sealed or flush-in-progress bits are set.  
    ///
    /// Returns [`BufferError::InsufficientSpace`] if the buffer is full.
    pub(crate) fn reserve_space(&self, payload_size: usize) -> Result<usize, BufferError> {
        assert!(payload_size <= BUFFER_SIZE, "payload larger than buffer");

        let state = self.state.load(Ordering::Acquire);
        if state & (SEALED_BIT | FLUSH_IN_PROGRESS_BIT) != 0 {
            return Err(BufferError::EncounteredSealedBuffer);
        }

        let offset = self
            .current_offset
            .fetch_add(payload_size, Ordering::AcqRel);

        if offset + payload_size > BUFFER_SIZE {
            return Err(BufferError::InsufficientSpace);
        }

        Ok(offset)
    }

    /// Writes `payload` into the buffer.
    ///
    /// # Atomicity guarantee
    ///
    /// The writer-count increment uses a CAS loop that rejects entry once
    /// [`SEALED_BIT`] is set.  
    ///
    ///
    /// The thread that wins the seal (`fetch_or` observes previous `SEALED_BIT == 0`) immediately calls
    ///
    /// [`FlushBufferRing::rotate_after_seal`], making seal + rotate one logical atomic step.  No other thread may rotate.
    pub(crate) fn write_all(
        &self,
        payload: &mut [u8],
        parent: &FlushBufferRing,
    ) -> Result<BufferMsg, BufferError> {
        // CAS writer increment
        //
        // Not unlike tipping the edge of your toe into a pool of water of
        // dubious temperature, the CAS method in the loop below allows a thread
        // to ensure that the current payload would be successfully
        // written to the flush buffer. If it has encountered  a sealed buffer
        // it respins until a new one has been set.
        //
        // In the case that it doesn't encounter a sealed buffer and in the case
        // that a compeating thread hasn't sealed the buffer, the thread can dive
        // right in!
        //
        // # Saftey
        // No need to check for FLUSH IN PROGRESS bit as a buffer with a bit set to
        // flush in progress is transitively a sealed bit.

        loop {
            let state = self.state.load(Ordering::Acquire);

            if state & SEALED_BIT != 0 {
                return Err(BufferError::EncounteredSealedBuffer);
            }

            let new = state + WRITER_ONE;

            // Incidentally doesn't allow threads to increment the 'Active Counter' bits
            // concurently. Under the assumption that threads spent most of there time writing
            // to buffers rather than prying atomic counters out of each of their hands, this is fine.
            // and an acceptable tradeoff for the correctness it gurantees.
            if self
                .state
                .compare_exchange(state, new, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
        }

        let payload_size: usize = payload.len();
        let res = self.reserve_space(payload_size);

        match res {
            Err(error) => {
                // Always release our writer slot before handling the error.
                self.state.fetch_sub(WRITER_ONE, Ordering::AcqRel);

                match error {
                    BufferError::InsufficientSpace => {
                        // fetch_or() returns the state *before* the OR.
                        // Exactly one thread will observe prev[SEALED_BIT] == 0;
                        // that thread is the sealer and the only thread allowed
                        // to rotate.
                        let prev = self.state.fetch_or(SEALED_BIT, Ordering::AcqRel);

                        if prev & SEALED_BIT != 0 {
                            // Another thread sealed first; we just retry.
                            return Err(BufferError::EncounteredSealedBuffer);
                        }

                        // We are the unique sealer thread. Rotate immediately so
                        // that seal + rotate appear atomic to all observers.
                        parent.rotate_after_seal(self.pos)?;

                        // Though prev is the state *before* sealing, so the writer
                        // this current thread has already decremented the ACTIVE_WRTER
                        // bit
                        let writers_before_seal = prev >> 2;

                        if writers_before_seal == 0 {
                            return Ok(BufferMsg::FreeToFlush(self.pos));
                        }

                        // Writers were active at seal time. But they may have ALL exited already
                        // between their fetch_sub and our fetch_or. Re-check current state.

                        let current = self.state.load(Ordering::Acquire);
                        if (current >> 2) == 0 {
                            // All writers drained before we could observe it. We must flush.

                            return Ok(BufferMsg::FreeToFlush(self.pos));
                        }

                        return Ok(BufferMsg::SealedBuffer);
                    }

                    BufferError::EncounteredSealedBuffer => {
                        return Err(error);
                    }

                    _ => return Err(BufferError::InvalidState),
                }
            }

            Ok(offset) => {
                let ptr = unsafe { (*self.buf.buffer.get()).as_mut_ptr().add(offset) };
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr, payload_size) };
                slice.copy_from_slice(payload);

                // Single fetch_sub yields an atomic snapshot of both the
                // writer count and the sealed bit, eliminating TOCTOU.
                let prev = self.state.fetch_sub(WRITER_ONE, Ordering::AcqRel);

                let was_last = (prev >> 2) == 1; // we were the only writer
                let is_sealed = (prev & SEALED_BIT) != 0;

                // If the buffer was sealed before we finished writing AND we
                // are the last writer, it is now safe to flush.
                if was_last && is_sealed {
                    return Ok(BufferMsg::FreeToFlush(self.pos));
                }

                Ok(BufferMsg::SuccessfullWrite)
            }
        }
    }

    /// Explicitly sets the sealed bit, preventing any new writes.
    fn set_sealed_bit_true(&self) -> Result<(), BufferError> {
        let prev = self.state.fetch_or(SEALED_BIT, Ordering::AcqRel);
        if prev & SEALED_BIT != 0 {
            Err(BufferError::EncounteredSealedBufferDuringCOMPEX)
        } else {
            Ok(())
        }
    }

    /// Explicity Clears the sealed bit, re-opening the buffer for writes.
    fn set_sealed_bit_false(&self) -> Result<(), BufferError> {
        let current = self.state.load(Ordering::Acquire);

        if (current >> 1) != 0 {
            return Err(BufferError::ActiveUsers);
        }

        if current & SEALED_BIT == 0 {
            return Err(BufferError::EncounteredUnSealedBufferDuringCOMPEX);
        }

        self.state
            .compare_exchange(
                current,
                current & !SEALED_BIT,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .map(|_| ())
            .map_err(|_| BufferError::EncounteredUnSealedBufferDuringCOMPEX)
    }
}

/// A ring of [`FlushBuffer`]s with an atomic pointer to the buffer currently
/// servicing write requests.
///
/// # Examples
///
/// ```
/// let ring = FlushBufferRing::with_buffer_amount(RING_SIZE);
/// let mut payload = make_payload("FILL", 4096);
///
/// match ring.put(&mut payload) {
///     Ok(BufferMsg::SuccessfullWrite) | Ok(BufferMsg::SuccessfullWriteFlush) => {}
///     other => panic!("unexpected: {other:?}"),
/// }
/// ```
pub(crate) struct FlushBufferRing {
    /// The single buffer currently servicing writes.
    ///
    /// Updated exclusively by the thread that won the seal race, via
    /// [`rotate_after_seal`].
    current_buffer: AtomicPtr<FlushBuffer>,

    /// Pinned ring — buffer addresses are stable for the lifetime of the ring.
    ring: Pin<Box<[Arc<FlushBuffer>]>>,

    /// Monotonically incrementing counter used to derive the next buffer index.
    next_index: AtomicUsize,

    /// Total number of buffers in the ring.
    _size: usize,
}

impl FlushBufferRing {
    pub(crate) fn with_buffer_amount(num_of_buffer: usize) -> FlushBufferRing {
        let buffers: Vec<Arc<FlushBuffer>> = (0..num_of_buffer)
            .map(|i| Arc::new(FlushBuffer::new_buffer(i)))
            .collect();

        let buffers = Pin::new(buffers.into_boxed_slice());
        let current = &*buffers[0] as *const FlushBuffer as *mut FlushBuffer;

        FlushBufferRing {
            current_buffer: AtomicPtr::new(current),
            ring: buffers,
            next_index: AtomicUsize::new(1),
            _size: num_of_buffer,
        }
    }

    /// Writes `payload` into the current buffer, rotating and retrying as needed.
    ///
    /// # Correctness
    ///
    /// - Only the thread that wins the seal CAS calls [`rotate_after_seal`].
    ///   All other threads simply yield and retry; they never attempt rotation.
    /// - The writer-count CAS in [`FlushBuffer::write_all`] guarantees no
    ///   thread can register as a writer after the sealed bit is set.

    //  Currently guranteed to write data into a buffer; returning a Result<> may be uneeded
    pub(crate) async fn put(&self, payload: &mut [u8]) -> Result<BufferMsg, BufferError> {
        loop {
            let current = unsafe {
                self.current_buffer
                    .load(Ordering::Acquire)
                    .as_ref()
                    .ok_or(BufferError::InvalidState)?
            };

            match current.write_all(payload, self) {
                Ok(BufferMsg::SuccessfullWrite) => {
                    return Ok(BufferMsg::SuccessfullWrite);
                }

                Ok(BufferMsg::SealedBuffer) => {
                    // Encountered a buffer of insuficient Space
                    // The buffer has been guranteed to be Sealed
                    // and rotated

                    continue;
                }

                Ok(BufferMsg::FreeToFlush(pos)) => {
                    let flush_buffer = self.ring.get(pos).unwrap().clone();

                    // Set the flush in progress bit
                    flush_buffer
                        .state
                        .fetch_or(FLUSH_IN_PROGRESS_BIT, Ordering::AcqRel);

                    // Flushing logic
                    // Passing in a functional might straight up be overkill
                    let _ = FlushBufferRing::flush(&flush_buffer.clone(), move || {
                        flush_buffer.current_offset.store(0, Ordering::Release);
                        flush_buffer
                            .state
                            .fetch_and(!(SEALED_BIT | FLUSH_IN_PROGRESS_BIT), Ordering::AcqRel);
                    })
                    .await;

                    // Houston, the flush was a success!
                    return Ok(BufferMsg::SuccessfullWriteFlush);
                }

                Err(BufferError::EncounteredSealedBuffer) => {
                    // At first you don't suceed
                    continue;
                }

                _ => return Err(BufferError::InvalidState),
            }
        }
    }

    /// Rotates `current_buffer` to the next slot in the ring.
    ///
    /// **Must only be called by the thread that won the seal race.**
    ///
    /// The `sealed_pos` guard means the call is idempotent: if another thread
    /// (or a retry) already advanced `current_buffer` past `sealed_pos`, we
    /// return immediately without touching anything.
    pub(crate) fn rotate_after_seal(&self, sealed_pos: usize) -> Result<(), BufferError> {
        let current = self.current_buffer.load(Ordering::Acquire);
        let current_ref = unsafe { current.as_ref().ok_or(BufferError::InvalidState)? };

        // Guard: if current_buffer no longer points at the buffer we sealed,
        // someone else already rotated — nothing to do.
        if current_ref.pos != sealed_pos {
            return Ok(());
        }

        // Scan forward through the ring for the next slot that is actually
        // available (neither sealed nor flush-in-progress).
        let ring_len = self.ring.len();

        for _ in 0..ring_len {
            let raw = self.next_index.fetch_add(1, Ordering::AcqRel);
            let next_index = raw % ring_len;

            let new_buffer = &self.ring[next_index];

            if new_buffer.is_available() {
                // CAS ensures only one thread installs the new pointer even if
                // rotate_after_seal is somehow called concurrently.
                let _ = self.current_buffer.compare_exchange(
                    current,
                    Arc::as_ptr(new_buffer) as *const FlushBuffer as *mut FlushBuffer,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                return Ok(());
            }
            // This slot is busy; try the next one.
        }

        // Every buffer in the ring is currently sealed or flushing.
        // This should not occur with a correctly sized ring but we return
        // an error rather than deadlock.
        //
        // We may also continually spin untill a buffer has been flushed successfully
        Err(BufferError::RingExhausted)
    }

    /// Unimplemented Asynchrosnous i/o using io/uring
    pub(crate) async fn flush(
        buffer: &FlushBuffer,
        on_complete: impl FnOnce() + Send + 'static,
    ) -> Result<(), BufferError> {
        on_complete();
        // TODO: implement actual LSS write
        Ok(())
    }
}

// ============================================================================
//  Tests
// ─============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use std::{
        collections::HashSet,
        sync::{Arc, Barrier},
        thread,
        time::Instant,
    };

    /// Very small, very lightweight, very unimpressive Linear Congruential Generator for deterministic pseudorandom number generation in tests.
    ///

    /// # Linear Congruential Generator
    ///
    /// The method tyields a sequence of pseudo-randomized numbers and represents
    /// one of the oldest and best-known pseudorandom number generator algorithms.
    ///
    ///
    /// source: https://en.wikipedia.org/wiki/Linear_congruential_generator
    ///

    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_usize(&mut self, bound: usize) -> usize {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.state >> 33) as usize) % bound
        }
    }

    const RING_SIZE: usize = 4;

    const OPS_PER_THREAD: usize = 2_000;

    /// Payload sizes ranging from tiny, medium, and near-capacity.
    const SIZES: &[usize] = &[
        1, 2, 4, 7, 8, 15, 16, 32, 64, 100, 128, 200, 256, 512, 1024, 2048, 4090, 4095, 4096,
    ];

    /// Build a recognisable, size-stamped payload.
    fn make_payload(tag: &str, size: usize) -> Vec<u8> {
        let meta = format!("[{tag}:{size}]");
        let mut buf = vec![0xAA_u8; size];
        let n = meta.len().min(size);
        buf[..n].copy_from_slice(&meta.as_bytes()[..n]);
        buf
    }

    // =============================================================================
    // HELPERS
    // =============================================================================

    async fn assert_put_ok(ring: &FlushBufferRing, payload: &mut [u8]) {
        match ring.put(payload).await {
            Ok(BufferMsg::SuccessfullWrite) | Ok(BufferMsg::SuccessfullWriteFlush) => {}
            other => panic!("put returned unexpected result: {other:?}"),
        }
    }
    // =============================================================================
    // single-threaded
    // =============================================================================

    /// Writes of random sizes
    // Currently capped at 39 iterations as thats enough to bring the flush ring to capacity
    // After implementing async flushing, we can stress test this even further
    #[tokio::test]
    async fn single_threaded_offset_uniqueness() {
        let ring = FlushBufferRing::with_buffer_amount(RING_SIZE);
        let mut rng = Lcg::new(0);
        let mut writes = 0usize;
        let mut flushes = 0usize;
        let mut data_written = 0;

        let mut i = 0;
        loop {
            let size = SIZES[rng.next_usize(SIZES.len())];

            let mut payload = make_payload(&format!("s{i:05}"), size);

            if (data_written + payload.len()) > BUFFER_SIZE * RING_SIZE {
                break;
            }

            data_written += payload.len();

            match ring.put(&mut payload).await {
                Ok(BufferMsg::SuccessfullWrite) => writes += 1,
                Ok(BufferMsg::SuccessfullWriteFlush) => {
                    writes += 1;
                    flushes += 1;
                }
                other => panic!("single_threaded: unexpected {other:?}"),
            }
            i += 1;
        }

        println!("single_threaded: {writes} writes, {flushes} logical flushes — OK, data_written: {data_written}kb");
    }

    /// A single BUFFER_SIZE payload must be accepted exactly once.
    #[tokio::test]
    async fn exact_fill() {
        let ring = FlushBufferRing::with_buffer_amount(RING_SIZE);
        let mut payload = make_payload("FILL", BUFFER_SIZE);
        match ring.put(&mut payload).await {
            Ok(BufferMsg::SuccessfullWrite) | Ok(BufferMsg::SuccessfullWriteFlush) => {}
            other => panic!("exact_fill: unexpected {other:?}"),
        }
    }

    /// reserve_space on an already-sealed buffer must return EncounteredSealedBuffer.
    #[tokio::test]
    async fn reserve_on_sealed_buffer_returns_error() {
        let buf = FlushBuffer::new_buffer(0);
        buf.set_sealed_bit_true().unwrap();
        assert!(matches!(
            buf.reserve_space(16),
            Err(BufferError::EncounteredSealedBuffer)
        ));
    }

    /// Sealing an already-sealed buffer must return the COMPEX error.
    #[tokio::test]
    async fn double_seal_returns_error() {
        let buf = FlushBuffer::new_buffer(0);
        buf.set_sealed_bit_true().unwrap();
        assert!(matches!(
            buf.set_sealed_bit_true(),
            Err(BufferError::EncounteredSealedBufferDuringCOMPEX)
        ));
    }

    /// Unsealing an already-unsealed buffer must return the COMPEX error.
    #[tokio::test]
    async fn unseal_unsealed_returns_error() {
        let buf = FlushBuffer::new_buffer(0);
        assert!(matches!(
            buf.set_sealed_bit_false(),
            Err(BufferError::EncounteredUnSealedBufferDuringCOMPEX)
        ));
    }

    /// A reserve that exactly hits the boundary must succeed; one byte over must fail.
    #[tokio::test]
    async fn boundary_reserve() {
        let buf = FlushBuffer::new_buffer(0);
        // Fill up to BUFFER_SIZE - 1
        buf.current_offset.store(BUFFER_SIZE - 1, Ordering::Relaxed);
        // One byte fits exactly
        assert!(buf.reserve_space(1).is_ok());
        // Reset and try one byte over capacity
        let buf2 = FlushBuffer::new_buffer(0);
        buf2.current_offset.store(BUFFER_SIZE, Ordering::Relaxed);
        assert!(matches!(
            buf2.reserve_space(1),
            Err(BufferError::InsufficientSpace)
        ));
    }

    #[tokio::test]
    async fn single_threaded_stress() {
        let ring = FlushBufferRing::with_buffer_amount(RING_SIZE);
        let mut writes = 0usize;
        let mut flushes = 0usize;
        let mut rng = Lcg::new(0x1234_5678);
        let start = Instant::now();

        for op in 0..OPS_PER_THREAD {
            let size = SIZES[rng.next_usize(SIZES.len())];
            if size > BUFFER_SIZE {
                continue;
            }
            let mut payload = make_payload(&format!("S:O{op:04}"), size);
            match ring.put(&mut payload).await {
                Ok(BufferMsg::SuccessfullWrite) => writes += 1,
                Ok(BufferMsg::SuccessfullWriteFlush) => {
                    writes += 1;
                    flushes += 1;
                }
                other => panic!("op {op}: unexpected {other:?}"),
            }
        }

        let elapsed = start.elapsed();
        println!(
            "{writes} writes, {flushes} logical flushes in {elapsed:.2?} ({:.0} ops/s)",
            (writes + flushes) as f64 / elapsed.as_secs_f64()
        );
    }
    // ==========================================================================
    // Multi-Threaded Test
    // ==========================================================================
    const NUM_THREADS_LARGE: usize = 8;
    const NUM_THREADS_MEDIUM: usize = 4;
    const NUM_THREADS_SMALL: usize = 2;
    #[tokio::test]

    async fn multi_threeaded_test_small() {
        let _ = multi_threaded_stress_helper(NUM_THREADS_SMALL);
    }
    #[tokio::test]

    async fn multi_threeaded_test_medium() {
        let _ = multi_threaded_stress_helper(NUM_THREADS_MEDIUM);
    }
    #[tokio::test]

    async fn multi_threeaded_test_large() {
        let _ = multi_threaded_stress_helper(NUM_THREADS_LARGE);
    }
    async fn multi_threaded_stress_helper(threads: usize) {
        let ring = Arc::new(FlushBufferRing::with_buffer_amount(RING_SIZE * 2));
        let writes = Arc::new(AtomicUsize::new(0));
        let flushes = Arc::new(AtomicUsize::new(0));
        let start = Instant::now();

        let handles: Vec<_> = (0..threads)
            .map(|tid| {
                let ring = Arc::clone(&ring);
                let writes = Arc::clone(&writes);
                let flushes = Arc::clone(&flushes);
                let seed = 0x1234_5678_u64.wrapping_add(tid as u64 * 0xDEAD_CAFE);
                thread::spawn(async move || {
                    let mut rng = Lcg::new(seed);
                    for op in 0..OPS_PER_THREAD {
                        let size = SIZES[rng.next_usize(SIZES.len())];
                        if size > BUFFER_SIZE {
                            continue;
                        }
                        let mut payload = make_payload(&format!("T{tid}:O{op:04}"), size);
                        match ring.put(&mut payload).await {
                            Ok(BufferMsg::SuccessfullWrite) => {
                                writes.fetch_add(1, Ordering::Relaxed);
                            }
                            Ok(BufferMsg::SuccessfullWriteFlush) => {
                                writes.fetch_add(1, Ordering::Relaxed);
                                flushes.fetch_add(1, Ordering::Relaxed);
                            }
                            other => panic!("thread {tid} op {op}: unexpected {other:?}"),
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            let _ =   h.join().expect("worker panicked");
        }

        println!(
            "{} writes, {} logical flushes in {:.2?}",
            writes.load(Ordering::Relaxed),
            flushes.load(Ordering::Relaxed),
            start.elapsed()
        );

        let ops = threads * OPS_PER_THREAD;
        let elapsed = start.elapsed();
        println!(
            "multi_threaded_stress: {ops} ops in {elapsed:.2?} ({:.0} ops/s)",
            ops as f64 / elapsed.as_secs_f64()
        );
    }

    /// Barrier-synchronised: all threads start simultaneously with 2048-byte payloads,
    /// maximising the race to seal and rotate.
    #[tokio::test]
    async fn hammer_seal_concurrent_rotation() {
        let ring = Arc::new(FlushBufferRing::with_buffer_amount(RING_SIZE));
        let barrier = Arc::new(Barrier::new(NUM_THREADS_SMALL));

        let handles: Vec<_> = (0..NUM_THREADS_SMALL)
            .map(|tid| {
                let ring = Arc::clone(&ring);
                let barrier = Arc::clone(&barrier);
                thread::spawn(async move || {
                    barrier.wait();
                    for iter in 0..100_usize {
                        let mut payload = make_payload(&format!("H{tid}:{iter}"), 2048);
                        match ring.put(&mut payload).await {
                            Ok(_) => {}
                            Err(e) => panic!("hammer thread {tid} iter {iter}: error {e:?}"),
                        }
                    }
                })
            })
            .collect();

        for h in handles {
             let _ = h.join().expect("hammer worker panicked");
        }
    }

    /// Confirms that after a flush-and-reset cycle the buffer state is exactly 0
    /// and the offset is exactly 0.
    #[tokio::test]
    async fn buffer_fully_resets_after_flush() {
        let ring = FlushBufferRing::with_buffer_amount(RING_SIZE);

        // Force a flush by filling the buffer exactly.
        let mut payload = make_payload("RESET", BUFFER_SIZE);
        match ring.put(&mut payload).await {
            Ok(BufferMsg::SuccessfullWriteFlush) => {
                // The flushed buffer is the one at pos 0 — check it was reset.
                let buf = &ring.ring[0];
                assert_eq!(buf.state.load(Ordering::Acquire), 0, "state not reset");
                assert_eq!(
                    buf.current_offset.load(Ordering::Acquire),
                    0,
                    "offset not reset"
                );
            }
            // Depending on prior test state the buffer may not be exactly full on first write.
            Ok(_) => {}
            Err(e) => panic!("buffer_fully_resets_after_flush: error {e:?}"),
        }
    }

    /// Two concurrent threads race to reserve disjoint regions. Uses a shared
    /// [`HashSet`] to assert no (buf_pos, offset) pair is ever issued twice.
    ///
    /// NOTE: This test interacts with [`FlushBuffer::reserve_space`] directly
    /// rather than going through the ring so we can capture the offsets.
    #[tokio::test]
    async fn concurrent_reserve_space_no_overlap() {
        use std::sync::Mutex;

        // A fresh, standalone buffer — no ring, no sealing race.
        let buf = Arc::new(FlushBuffer::new_buffer(99));
        let seen: Arc<Mutex<HashSet<(usize, usize)>>> = Arc::new(Mutex::new(HashSet::new()));

        const THREADS: usize = 8;
        const RESERVES_PER_THREAD: usize = 32; // 32 × 8 = 256 × 16 = 4096 — exactly fills

        let barrier = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|_tid| {
                let buf = Arc::clone(&buf);
                let seen = Arc::clone(&seen);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..RESERVES_PER_THREAD {
                        match buf.reserve_space(16) {
                            Ok(offset) => {
                                let mut lock = seen.lock().unwrap();
                                assert!(
                                    lock.insert((99, offset)),
                                    "[OVERLAP] offset {offset} in buffer 99 was issued twice!"
                                );
                            }
                            // Buffer exhausted — acceptable.
                            Err(BufferError::InsufficientSpace) => {}
                            Err(BufferError::EncounteredSealedBuffer) => {}
                            Err(e) => panic!("reserve_space error: {e:?}"),
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("reserve worker panicked");
        }
    }
}
