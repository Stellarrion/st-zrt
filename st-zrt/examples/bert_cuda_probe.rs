//! CUDA placement probe for BERT-style text encoder ONNX graphs.
//!
//! Usage:
//!   cargo run -p st-zrt --example bert_cuda_probe --features cuda -- \
//!     /path/to/model_cuda.onnx [batch] [seq] [hidden]

use st_zrt::{
    CudaPreset, Environment, GraphOptimizationLevel, LoggingLevel, MemoryInfo, OwnedValue,
    RunInput, Session, SessionOptions, StaticIoLane, Tensor,
};
use std::sync::Arc;

fn main() -> st_zrt::Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("model path required");
    let batch = args
        .next()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(64);
    let seq = args
        .next()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(32);
    let hidden = args
        .next()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(384);

    let env = Environment::new_with_level(LoggingLevel::Verbose, "bert-cuda-probe")?;
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_log_id("bert-cuda-probe")?
        .with_log_severity(LoggingLevel::Verbose)
        .with_log_verbosity(1)
        .with_intra_threads(1)
        .with_inter_threads(1)
        .with_config_entry("session.use_device_allocator_for_initializers", "1")
        .expect("config entry")
        .with_cuda_preset(CudaPreset::performance(0))?;
    let sess = Arc::new(Session::new(&env, &model, opts)?);

    let mem = MemoryInfo::cpu()?;
    let len = batch * seq;
    let attention_mask = vec![1_i64; len];
    let token_type_ids = vec![0_i64; len];

    let shape = [batch as i64, seq as i64];
    let output_shape = [batch as i64, seq as i64, hidden as i64];
    for seed in [0_i64, 1000] {
        let input_ids = input_ids_pattern(batch, seq, seed);
        let inputs = [
            Tensor::from_buffer(&input_ids, &shape, &mem)?,
            Tensor::from_buffer(&attention_mask, &shape, &mem)?,
            Tensor::from_buffer(&token_type_ids, &shape, &mem)?,
        ];
        let input_refs = inputs
            .iter()
            .map(|input| input as &dyn RunInput)
            .collect::<Vec<_>>();
        let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
        sess.run(&input_refs, &mut out)?;
        let output = out[0].as_ref().expect("output[0]");
        println!(
            "session-run seed={seed} batch={batch} seq={seq}; output memory={:?}; len={}",
            output.memory_info().ok(),
            output.as_slice::<f32>().map(|s| s.len()).unwrap_or(0)
        );
        print_prefixes("session-run", output.as_slice::<f32>()?, batch, seq, hidden);
    }

    let mut lane = StaticIoLane::<i64, f32, 3, 1>::new(
        sess.clone(),
        &mem,
        [&shape, &shape, &shape],
        [&output_shape],
    )?;
    lane.set_rebind_inputs_each_run(true);
    lane.input_mut(1)?.copy_from_slice(&attention_mask);
    lane.input_mut(2)?.copy_from_slice(&token_type_ids);
    for seed in [0_i64, 1000] {
        let input_ids = input_ids_pattern(batch, seq, seed);
        lane.input_mut(0)?.copy_from_slice(&input_ids);
        lane.run()?;
        println!("static-lane seed={seed} batch={batch} seq={seq}");
        print_prefixes("static-lane", lane.output(0)?, batch, seq, hidden);
    }
    Ok(())
}

fn input_ids_pattern(batch: usize, seq: usize, seed: i64) -> Vec<i64> {
    let mut input_ids = vec![0_i64; batch * seq];
    for row in 0..batch {
        input_ids[row * seq] = 101;
        for col in 1..seq.saturating_sub(1) {
            input_ids[row * seq + col] = 2023 + seed + ((row + col) as i64 % 128);
        }
        if seq > 1 {
            input_ids[row * seq + seq - 1] = 102;
        }
    }
    input_ids
}

fn print_prefixes(label: &str, values: &[f32], batch: usize, seq: usize, hidden: usize) {
    for row in 0..batch.min(2) {
        let start = row * seq * hidden;
        let end = (start + 8).min(values.len());
        println!(
            "{label} row {row} first token prefix {:?}",
            &values[start..end]
        );
    }
}
