//! st-zrt A/B benchmark harness (M0 — the feasibility gate).
//!
//! Three variants share one libonnxruntime:
//!  - A: ort default (copying path)
//!  - B: ort expert  (IoBinding + prealloc + RunOptions reuse)
//!  - C: st-zrt proto (pre-marshal + genuine zero-copy)
#![allow(dead_code)]

pub mod micro;
pub mod models;
pub mod ort_default;
pub mod ort_expert;
