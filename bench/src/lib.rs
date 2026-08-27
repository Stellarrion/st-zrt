//! ortx benchmark harness (M0 — the feasibility gate).
//!
//! See ../BENCHMARK.md. Three variants share one libonnxruntime:
//!  - A: ort default (copying path)
//!  - B: ort expert  (IoBinding + prealloc + RunOptions reuse)
//!  - C: ortx proto  (pre-marshal + genuine zero-copy)  [task #6]
#![allow(dead_code)]

pub mod micro;
pub mod models;
pub mod ort_default;
pub mod ort_expert;
