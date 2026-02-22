//! Filepath: src/lib.rs
//!
//! # LLAMA
//!
//! A general-purpose, high-performance concurrent storage and caching subsystem
//! designed for modern hardware.
//!
//! LLAMA is intentionally architected to remain independent of transactional
//! database functionality and specific access method implementations.
//!
//! By leveraging the performance characteristics of append-only writes on flash
//! storage avoidiing random writes, and amortizing cost via large multi-page buffers.
//! 
//! LLAMA enables access methods to achieve both high throughput and
//! latch-free operation through its interfaces.
//!
//! To support structural modifications, LLAMA provides a limited form of
//! system-level transactions. These allow access methods to safely 'grow' and
//! 'shrink' their structures. Eg. handling page splits and merges in
//! B-tree–like data structures.
//!
//! 
//! # Design
//! 
//! LLAMA is comprised of:
//!     Mapping Tables: In memory lock-free map to locate page deltas on both in memory in the caching layer as well on Disks whethre remote or local.
//! 
//!     Caching Layer : The physically store hot deltas
//! 
//!     Log-Structured Secondary Storage : To provides the usual advantages of avoiding random writes, reducing
//!                                        the number of writes via large multi-page buffers, and wear leveling  
//!                                        needed by flash memory
//! 
//!     Recovery Protocal: To rebuild Mapping Table after crashes
//! 
//! 

pub mod flush_buffer;
pub mod data_objects;
