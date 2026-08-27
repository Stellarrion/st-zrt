# st-zrt Completeness Audit

> **Status: HISTORICAL (2026-06-23).** This audit froze against `st-zrt-sys 1.27.0` and the 0.2/0.3
> API surface. The counts, tier tables, and unwrapped-accessor lists below describe that snapshot
> and are kept as evidence, not as a live claim: the current tree (`st-zrt-sys 1.27.1`, `st-zrt`
> 0.3.0) has since wrapped additional accessors (EP authoring/introspection, profiling-event
> revisions) and removed others. Do not update this document piecemeal; re-run the methodology
> (below) against HEAD if a current coverage number is needed.

**Date:** 2026-06-23 · **ORT:** 1.27.0 (`st-zrt-sys` 1.27.0, API version 27, 422 `OrtApi` accessors)

A systematic audit of how much of ONNX Runtime's C API the st-zrt safe layer actually wraps, what is
deliberately left raw, and what the genuine gaps are — as a roadmap to "complete."

## Implementation progress

Working through Tier 1 → Tier 2 one theme at a time (implement TDD → review → re-audit coverage → advance).
Each item flips its accessors from "unused" to "used" in the methodology grep.

| # | Theme | Accessors | Status |
|---|---|--:|---|
| T1.1 | LoRA adapters | 4 | ✅ done |
| T1.2 | Model-compat pre-flight | 3 | ✅ done |
| T1.3 | External/file-backed initializers | 7 | ✅ done |
| T1.4 | SyncStream | 4 | ✅ wrapper (CUDA happy-path test deferred) |
| T1.5 | RunOptions profiling | 4 | ✅ done |
| T1.6 | Custom logger | 7 | ✅ done |
| T1.7 | String-tensor element access | 4 | ✅ done |
| T1.8 | Hardware device enumeration | 2 (+5 device-incompat details folded here) | ✅ done |
| T2.9–T2.20 | Tier 2 completeness | 77 | T2.9✅ T2.10✅ T2.11✅ T2.12✅ T2.13✅ T2.14✅ T2.15✅ T2.16✅ T2.17✅ T2.18✅ T2.19✅ T2.20✅ |

**Wrapped tally:** 211 → 344 / 422 (82%).

## TL;DR

- **422** FFI accessors generated; **211 wrapped (50%)** by the safe layer; **211 unwrapped.**
- Of the 211 unwrapped: **68 are deliberate non-goals** (raw by design), **31 are covered via an
  alternative path** (not gaps), and **~112 are genuine gaps** worth wrapping.
- Genuine gaps cluster into a **Tier 1 serving-critical** set (≈35 accessors) and a **Tier 2
  completeness** set (≈77). See [Recommended roadmap](#recommended-roadmap).

## Methodology (reproducible)

The audit compares every accessor defined in `st-zrt-sys/src/generated.rs` against every accessor
*invoked* anywhere in `st-zrt/src/*.rs`. An accessor counts as "wrapped" if it appears as a method call
`.NAME(` in the safe layer (catches `api().NAME(`, `api.NAME(` where `api` is a local binding, and
`self.NAME(` alike):

```bash
# defined
grep -oE 'pub unsafe fn [a-z_0-9]+' st-zrt-sys/src/generated.rs \
  | sed 's/pub unsafe fn //' | sort -u > defined.txt   # 422

# used (robust: any .NAME( call site)
> used.txt
while read -r name; do
  grep -qE "\.$name\(" st-zrt/src/*.rs && echo "$name" >> used.txt
done < defined.txt                                      # 211

# unused = defined − used                               # 211
comm -23 defined.txt used.txt > unused.txt
```

> An earlier grep/sed pass undercounted by missing the `api.NAME(` local-binding form; the per-name
> `.NAME(` check above is the accurate one. (Spot-checked: `session_get_input_name`,
> `add_session_config_entry`, `enable_cpu_mem_arena`, `set_intra_op_num_threads`,
> `set_session_graph_optimization_level` are all confirmed **used**.)

## Coverage breakdown

| Bucket | Count | Verdict |
|---|--:|---|
| **Wrapped** (safe layer calls `.NAME(`) | 211 | done |
| **Non-goal — model-editor graph traversal** (`graph__`/`node__`/`value_info__`/`op_attr__`/`shape_infer_context__`) | 46 | raw via gateway; wrapped on-demand |
| **Wrapped 0.3.0 — EP introspection** (`ep_assigned`/`ep_device`/`hardware_device`/`session__get_ep_graph_assignment_info`) | 15 | borrowed views in `ep_device.rs`/`hardware.rs`/`session.rs`; full `OrtEpApi` helper module is Phase 3 |
| **Non-goal — sub-API gateways** (`get_training_api`, `get_execution_provider_api`) | 2 | raw gateway by design |
| **Non-goal — device-EP incompatibility internals** (`device_ep_incompatibility_details__*`) | 5 | (also feeds the compat gap) |
| **Alt path — typed provider-options structs** (`create/update/release/get_*_as_string/_by_name`) | 23 | st-zrt uses key/value string options |
| **Alt path — legacy EP append** (`session_options_append_execution_provider_*`) | 8 | options-struct append used instead |
| **Genuine gap — Tier 1 (serving-critical)** | 35 | see below |
| **Genuine gap — Tier 2 (completeness)** | 77 | see below |

---

## Tier 1 — serving-critical gaps (≈35)

The set that makes st-zrt "complete for a serving runtime" (rs-celer). Self-contained, testable on the
RTX 4090.

### 1. LoRA adapters — 4 ✅
`create_lora_adapter`, `create_lora_adapter_from_array`, `release_lora_adapter`,
`run_options_add_active_lora_adapter`.
**Implemented:** `st-zrt/src/lora.rs` (`LoraAdapter::from_path`/`from_array`, Drop=release,
process-singleton default allocator) + `RunOptions::add_active_lora_adapter`. Test:
`lora_adapter_from_array_rejects_invalid_bytes` (garbage → ORT error, clean drop).
LLM/BERT adapter serving — load a base model once, hot-swap LoRA weights per request. **Not wrapped at
all.** Highest serving value.

### 2. Model compatibility pre-flight (ORT 1.27) — 3 ✅
`get_compatibility_info_from_model`, `get_compatibility_info_from_model_bytes`,
`get_model_compatibility_for_ep_devices` (+ the 5 `device_ep_incompatibility_details__*`/`
get_hardware_device_ep_incompatibility_details` internals — folded into T1.8, they operate on
`HardwareDevice`).
**Implemented:** `st-zrt/src/compat.rs` — `compatibility_info_from_path`/`_bytes` (→
`Option<String>`; `None` = no compat info, i.e. a standard non-precompiled model, per the ORT
contract) + `CompiledModelCompatibility` enum (EP_NOT_APPLICABLE/SUPPORTED_OPTIMAL/
SUPPORTED_PREFER_RECOMPILATION/UNSUPPORTED) + `compatibility_for_ep_devices` (feature `ep`).
Test: `compatibility_info_preflight_before_load`.

### 3. External / file-backed initializers — 7 ✅
`add_external_initializers`, `add_external_initializers_from_files_in_memory`,
`create_external_initializer_info`, `release_external_initializer_info`,
`external_initializer_info__get_byte_size/_file_offset/_file_path`.
**Implemented:** `ExternalInitializerInfo` handle (`initializer.rs`: `new` + path/offset/size
getters, Drop=release) and three `Session` ctors — `new_with_external_initializers` (batch
`AddExternalInitializers`: replaces an initializer the model marks external-data with an in-memory
OrtValue, name/shape/dtype verified), `new_with_external_initializer_files`
(`AddExternalInitializersFromFilesInMemory`: supplies external `.data` file contents from memory),
plus the info round-trip. Fixture `external_add.onnx` (+`.data`) generated by
`tests/fixtures/gen_external_data.py`. Tests: `external_initializer_info_round_trips`,
`external_initializers_batch_replaces_external_backing`, `external_initializer_files_in_memory`.

### 4. SyncStream — 4 ✅ *(wrapper; CUDA happy-path test deferred)*
`create_sync_stream_for_ep_device`, `run_options_set_sync_stream`, `sync_stream__get_handle`,
`release_sync_stream`.
**Implemented:** `Arc<SyncStream>` owned wrapper in `ep_device.rs` (feature `ep`) —
`for_ep_device` (`CreateSyncStreamForEpDevice`, retaining the device's `Arc<EnvInner>` guard),
`native_handle` (`SyncStream_GetHandle`), `pub(crate) as_ptr` for stream-aware ORT APIs, and `Drop`
(`ReleaseSyncStream`). Pure `RunOptions::with_sync_stream` retains the Arc and transfers it to
`MaterializedRunOptions` before `RunOptionsSetSyncStream`, so the stream remains alive
across every run. Test: `ep_device::tests::sync_stream_construction_is_clean` (CPU EP → clean
`NOT_IMPLEMENTED` error path, proving the create FFI is reached). **CUDA happy-path test deferred:**
`cuda_ep.rs` uses the legacy `SessionOptionsAppendExecutionProvider` CUDA path, which does not
register an `OrtEpDevice`; exercising `native_handle`/owned stream attachment/`Drop` on a real CUDA stream
needs EP-library registration to surface a CUDA `EpDevice` via `get_ep_devices` (separate infra).
Design + safety contract per [`v0.2.2-sync-stream-plan.md`](./v0.2.2-sync-stream-plan.md).

### 5. RunOptions profiling — 4 ✅
`run_options_enable_profiling`, `run_options_disable_profiling`,
`run_options_get_run_log_severity_level`, `run_options_get_run_log_verbosity_level`.
**Implemented:** `RunOptions::enable_profiling`/`disable_profiling` (per-run Chrome-tracing toggle)
+ `log_severity`/`log_verbosity` getters (the counterparts to the existing setters). Test:
`run_options_profiling_and_log_getters_round_trip`.

### 6. Custom logger — 7 ✅
`create_env_with_custom_logger`, `create_env_with_custom_logger_and_global_thread_pools`,
`create_env_with_options`, `update_env_with_custom_log_level`, `set_user_logging_function`,
`logger__get_logging_severity_level`, `logger__log_message`.
**Implemented:**
- `Environment::new_with_logger` (`CreateEnvWithCustomLogger`) + `new_with_logger_and_global_thread_pools`
  (`CreateEnvWithCustomLoggerAndGlobalThreadPools`) route ORT logs into a caller `Fn(LogRecord)+Send+Sync`.
  The closure is boxed behind a `Sized` `LoggerSlot` (thin pointer — required to round-trip through
  ORT's `void* param` without losing the trait-object vtable), owned by `EnvInner` so it drops with the
  Env. A `log_trampoline` (correct `OrtLoggingFunction` signature) is `transmute`d to the generated
  `sys::LoggingFunction` alias, which **mislabels** the last two params as `status_messages`/`num_status_messages`
  (they are really `code_location`/`message` `const char*`; ABI-identical — asserted same size).
- `EnvCreationOptions` builder + `Environment::new_with_options` (`CreateEnvWithOptions`) mirror the
  `#[repr(C)]` `OrtEnvCreationOptions` value struct (version=`sys::API_VERSION`=27) field-for-field; the
  only path that also accepts environment-level config entries.
- `Environment::set_log_level` (`UpdateEnvWithCustomLogLevel`).
- `SessionOptions::with_user_logging_function` (`SetUserLoggingFunction`); the closure is leaked via
  `Arc::into_raw` (ORT retains it for the session's lifetime; no unregister API).
- `Logger<'a>` borrowed wrapper (feature `custom-ops`) over `OrtLogger*` from `KernelInfo`/`KernelContext`:
  `severity_level` (`Logger_GetLoggingSeverityLevel`) + `log` (`Logger_LogMessage`). The two `logger()`
  getters now return `Result<Option<Logger<'a>>>`.
**ORT caveat (documented):** the logging function is process-global (first `Env` wins). Tests:
`environment_with_custom_logger_*`, `env_creation_options_builder_*`, `environment_set_log_level_*`,
`environment_custom_logger_runs_session_without_ub`, `session_options_user_logging_function_*`; items 6–7
exercised via the logger calls now in `MyRelu::compute` (`custom_op_runs_end_to_end`).

### 7. String-tensor element access — 4 ✅
`fill_string_tensor_element`, `get_string_tensor_element`, `get_string_tensor_element_length`,
`get_resized_string_tensor_element_buffer`.
**Implemented:** `StringTensor::set_element` (`FillStringTensorElement`),
`set_element_utf8` (`GetResizedStringTensorElementBuffer`, raw UTF-8 bytes, no NUL), and
`element` (`GetStringTensorElementLength` + `GetStringTensorElement`). Test:
`string_tensor_element_access_round_trips`.

### 8. Hardware device enumeration — 2 (+5 device-incompat details) ✅
`get_hardware_devices`, `get_num_hardware_devices`, `get_hardware_device_ep_incompatibility_details`,
`release_device_ep_incompatibility_details`,
`device_ep_incompatibility_details__get_reasons_bitmask/_notes/_error_code`.
**Implemented:** `st-zrt/src/hardware.rs` — `num_hardware_devices`/`hardware_devices` (borrowed
`HardwareDevice` handles), `hardware_device_ep_incompatibility_details` (→ `Option<DeviceEpIncompatibilityDetails>`
with `reasons_bitmask`/`error_code`/`notes`, Drop=release). A CPU host enumerates ≥1 device; the
CPU EP yields (compatible, empty) details. Test: `hardware_enumeration_and_ep_incompatibility_details`.

---

## Tier 2 — completeness / ergonomics gaps (≈77)

### 9. Value / type / map / sequence / optional introspection — ~20 ✅
`has_value`, `is_tensor`, `get_type_info`, `get_onnx_type_from_type_info`, `get_denotation_from_type_info`,
`cast_type_info_to_{tensor,map,sequence,optional}_info`, `get_map_key_type`, `get_map_value_type`,
`release_map_type_info`, `get_sequence_element_type`, `release_sequence_type_info`,
`get_optional_contained_type_info`, `tensor_type_and_shape__has_shape`, `get_tensor_shape_element_count`,
`get_tensor_element_type_and_shape_data_reference`, `tensor_at`, `create_value`, `create_opaque_value`,
`get_opaque_value`, `get_value_info_name`, `get_value_info_type_info`.
**Implemented:** `OwnedValue` introspection (`is_tensor`, `has_value`, `type_info`,
`element_type_and_shape`, `tensor_at`, `new_sequence`/`new_map` via `create_value`, `opaque`/`get_opaque`);
`RuntimeTypeInfo::denotation` + the four casts returning **borrowed** views (`TensorTypeAndShapeInfoView`,
`MapTypeInfo`, `SequenceTypeInfo`, `OptionalTypeInfo`) with their element/key/value readers; TSI
`has_shape` + `shape_element_count`; `ValueInfo::name`/`type_info_kind` (model-editor). The map/seq/
optional casts are read-only (construction is impossible from the C API since 1.26). Test:
`owned_value_t2_9_introspection_round_trip` + `value_info_name_and_type_kind_round_trip`.
**Deliberate non-goals:** `release_map_type_info` / `release_sequence_type_info` — ORT documents the cast
outputs as borrowed ("Do not free this value"), so calling these would double-free. They remain unwrapped
by design (the only sound choice).

### 10. Key/value pairs API — 4 ✅
`create_key_value_pairs`, `add_key_value_pair`, `get_key_value`, `remove_key_value_pair`.
**Implemented:** `KeyValuePairs` owning wrapper (`allocator.rs`: `new`/`add`/`get`/`remove`,
Drop=`ReleaseKeyValuePairs`). Test: `key_value_pairs_round_trip`.

### 11. Allocator v2 / shared allocator / memory-info v2 — 9 ✅
`create_and_register_allocator_v2`, `create_shared_allocator`, `get_shared_allocator`,
`release_shared_allocator`, `create_memory_info_v2`, `compare_memory_info`, `allocator_get_info`,
`register_allocator`, `unregister_allocator`.
**Implemented:** `MemoryInfo::new_v2` (`CreateMemoryInfo_V2`, with new `MemoryInfoDeviceType` +
`DeviceMemoryType` enums) + `MemoryInfo::equals` (`CompareMemoryInfo`, polarity corrected: ORT writes
0 for equal), `Allocator::memory_info` (`AllocatorGetInfo` → snapshot),
`Environment::create_and_register_allocator_v2`, `register_existing_allocator` (`RegisterAllocator`,
idx 176 — the existing `register_allocator` wraps the v1 *create*-and-register), `unregister_allocator`,
and the ep-gated shared-allocator trio (`create_shared_allocator` / `get_shared_allocator` /
`release_shared_allocator`, CPU error-path tested).

### 12. Misc `SessionOptions` knobs — ~13 ✅
`clone_session_options`, `set_deterministic_compute`, `set_language_projection`,
`set_optimized_model_file_path`, `set_symbolic_dimensions`, `set_ep_dynamic_options`,
`session_options_set_load_cancellation_flag`, `set_per_session_thread_pool_callbacks`,
`session_options_set_custom_create_thread_fn`, `session_options_set_custom_join_thread_fn`,
`session_options_set_custom_thread_creation_options`, `session_options_set_ep_selection_policy_delegate`,
`get_session_execution_mode`, `get_mem_pattern_enabled`.
**Implemented:** spread across the types the FFI actually targets — `SessionOptions`
(`with_optimized_model_file_path`/`with_deterministic_compute`/`with_load_cancellation_flag` +
`execution_mode`/`mem_pattern_enabled` getters + `clone_ort_handle`, plus the expert
`with_custom_thread_handlers` (create/join/creation-options trio) and ep-gated
`with_ep_selection_policy_delegate`), `Environment` (`set_language_projection` +
`set_per_session_thread_pool_callbacks` with a `#[repr(C)]` `ThreadPoolCallbacksConfig` mirror of
the versioned v1.25 struct), `Session::set_ep_dynamic_options`, and
`TensorTypeAndShapeInfo::set_symbolic_dimensions`. The stock ORT release rejects per-session
thread-pool callbacks (`--enable_session_threadpool_callbacks` build flag); the test asserts that
documented error to prove the FFI path is sound.

### 13. Session-config introspection — 4 ✅
`get_session_config_entry`, `has_session_config_entry`, `get_session_options_config_entries`,
`get_run_config_entry`.
**Implemented:** `RunOptions::config_entry` (`GetRunConfigEntry`) + `SessionOptions::has_config_entry`
(`HasSessionConfigEntry`, via a transient built handle). The two that were deferred are now done:
`SessionOptions::config_entry` (`GetSessionConfigEntry` — two-call buffer-size dance) and
`SessionOptions::config_entries` (`GetSessionOptionsConfigEntries` → owning `KeyValuePairs` via a
new `KeyValuePairs::from_handle`). Test: `session_options_config_entry_round_trip`.

### 14. `RunOptions` extras — 3 ✅
`run_options_get_run_tag`, `run_options_unset_terminate`.
**Implemented:** `RunOptions::run_tag` (getter counterpart to `set_run_tag`) + `unset_terminate`
(counterpart to `terminate`). Test: `run_options_run_tag_and_unset_terminate`.

### 15. Overridable-initializer introspection — 3 ✅
`session_get_overridable_initializer_count`, `session_get_overridable_initializer_name`,
`session_get_overridable_initializer_type_info`.
**Implemented:** `Session::overridable_initializer_count`/`_name`/`_type_info`. The type-info
accessor returns an owning [`TypeInfo`] wrapper (`type_info.rs`) whose `onnx_type` uses
`GetOnnxTypeFromTypeInfo` (2 of T2.9 done early) and whose Drop uses `ReleaseTypeInfo`. Fixture
`overridable_add.onnx` (C as input+initializer → count 1). Test:
`session_overridable_initializers_introspect`.

### 16. Diagnostics — 5 ✅
`get_available_providers`, `release_available_providers`, `get_build_info_string`,
`get_current_gpu_device_id`, `set_current_gpu_device_id`.
**Implemented:** `st-zrt/src/diagnostics.rs` — `available_providers` (engine-allocated array,
`ReleaseAvailableProviders`-freed), `build_info` (static string), `current_gpu_device_id` /
`set_current_gpu_device_id`. Test: `diagnostics_providers_and_build_info`.

### 17. Telemetry — 2 ✅
`enable_telemetry_events`, `disable_telemetry_events`.
**Implemented:** `Environment::enable_telemetry_events`/`disable_telemetry_events`. Test:
`environment_telemetry_toggle`.

### 18. Custom-op extras — 6 ✅
`enable_ort_custom_ops`, `register_custom_ops_library`, `register_custom_ops_library_v2`,
`register_custom_ops_using_function`, `kernel_info_get_attribute_array_string`,
`kernel_info__get_config_entries`.
**Implemented:** `SessionOptions` custom-ops-gated builders `with_enable_ort_custom_ops`
(`EnableOrtCustomOps` — stock ORT errors without bundled onnxruntime-extensions; test asserts that),
`with_register_custom_ops_library` (v2, recommended), `with_register_custom_ops_library_v1` (legacy,
dlopen handle intentionally retained), `with_register_custom_ops_using_function`; and
`KernelInfo::attr_strings` (`KernelInfoGetAttributeArray_string` — allocator-freed array) +
`KernelInfo::config_entries` (`KernelInfo_GetConfigEntries` → `KeyValuePairs`). The two KernelInfo
reads are exercised through `MyRelu::create` in the end-to-end custom-op test.

### 19. EP-library registration — 2 ✅
`register_execution_provider_library`, `unregister_execution_provider_library`.
**Implemented:** `Environment::register_execution_provider_library` (load `.so` + register under a
name) + `unregister_execution_provider_library` (by name). Bogus-path / unknown-name reach the FFI
cleanly (test `environment_ep_library_registration_reaches_ffi`).

### 20. Misc — 4 ✅
`get_bound_output_names`, `create_tensor_with_data_and_deleter_as_ort_value`, `get_error_code`,
`get_error_message`.
**Implemented:** `IoBinding::output_names` (`GetBoundOutputNames` — contiguous non-NUL-terminated
UTF-8 buffer + parallel `lengths` array, both freed via the default allocator; test
`iobinding_output_names_round_trips`) and `OwnedValue::from_allocated` (`CreateTensorWithDataAndDeleterAsOrtValue`
— hands an `Allocator`-produced buffer to ORT, which frees it via that allocator on drop; extended
`Allocation` with `len`/`is_empty`/`allocator_handle`/`into_raw_parts`; test
`owned_value_from_allocated_buffer_is_sound`). `get_error_code`/`get_error_message` are already
surfaced at the sys layer (`status_to_result` → `Error{code,message}`), so no new wrapper is needed.

---

## Non-goals & alt-paths (do not chase)

- **Model-editor graph traversal** (`graph__*`/`node__*`/`value_info__*`/`op_attr__*`/
  `shape_infer_context__*`, 46) — available raw via the `model-editor` gateway; wrapped on demand.
  Completing this is a large surface with low serving value.
- **EP introspection** (`ep_assigned_*`, `ep_device__device/metadata/options/memory_info`,
  `hardware_device__type/vendor/vendor_id/device_id/metadata`, `session__get_ep_graph_assignment_info`,
  15) — **wrapped 0.3.0** as borrowed views (`EpDevice`/`HardwareDevice` accessors, `EpAssignedNode`/
  `EpAssignedSubgraph`, `Session::ep_graph_assignment_info`).
- **EP authoring** (`OrtEpApi` helper table, reached via the `get_execution_provider_api` gateway) —
  **wrapped 0.3.0** in `st-zrt/src/ep_authoring.rs` + `st-zrt-sys/src/ep_vtables.rs`: the implementor
  vtables (`EpVTable`/`EpFactoryVTable`/`NodeComputeInfoVTable`) + the independently-creatable helpers
  (`KernelDefBuilder`/`KernelDef`, `OpSchema`, `ProfilingEvent`) + the safe-Rust `EpAuthor`/
  `EpFactoryAuthor` traits, the `#[custom_ep]` cdylib macro, and in-process registration
  (`OwnedHardwareDevice`/`OwnedEpDevice`/`SessionOptions::append_ep_device`). Mirrors `custom_ops.rs`.
- **Typed provider-options structs** (23) — st-zrt deliberately uses the key/value string builder
  (`CudaProviderOptions` + `with_raw`), which is more future-proof than pinning `#[repr(C)]` option
  structs per provider. The `_as_string`/`_by_name` getters would add debug introspection only.
- **Legacy `append_execution_provider_*`** (8) — superseded by the options-struct append st-zrt uses.
- **Training** (`get_training_api`) — `OrtTrainingApi` is not in the CPU release headers; needs the
  `onnxruntime-training` package.

## Recommended roadmap

1. **Tier 1, items 1–3 first** (LoRA + compatibility pre-flight + external/file-backed initializers) —
   biggest serving value, self-contained, directly enable rs-celer use cases that currently cannot be
   expressed. Testable on the RTX 4090 (CUDA) + CPU.
2. **Tier 1, items 4–8** (SyncStream, profiling, custom logger, string-tensor elements, hardware
   enumeration) — rounds out the serving surface. SyncStream per its own plan.
3. **Tier 2** — value/type introspection + allocator v2 + SessionOptions knobs. Drives toward a
   "complete general wrapper."
4. **Keep raw** — model-editor graph traversal (on-demand), training (package), typed
   provider-options structs (alt path). *(EP-authoring, formerly listed here, is **done 0.3.0** — see
   above.)*

---

## Appendix A — all 211 unwrapped accessors (categorized)

**Model-editor graph traversal (46):** `graph__get_graph_view`, `graph__get_initializers`,
`graph__get_inputs`, `graph__get_model_metadata`, `graph__get_model_path`, `graph__get_name`,
`graph__get_nodes`, `graph__get_num_initializers`, `graph__get_num_inputs`, `graph__get_num_nodes`,
`graph__get_num_operator_sets`, `graph__get_num_outputs`, `graph__get_onnx_ir_version`,
`graph__get_operator_sets`, `graph__get_outputs`, `graph__get_parent_node`,
`node__get_attribute_by_name`, `node__get_attributes`, `node__get_domain`, `node__get_ep_name`,
`node__get_graph`, `node__get_id`, `node__get_implicit_inputs`, `node__get_inputs`, `node__get_name`,
`node__get_num_attributes`, `node__get_num_implicit_inputs`, `node__get_num_inputs`,
`node__get_num_outputs`, `node__get_num_subgraphs`, `node__get_operator_type`, `node__get_outputs`,
`node__get_since_version`, `node__get_subgraphs`, `op_attr__get_tensor_attribute_as_ort_value`,
`shape_infer_context__get_attribute`, `value_info__get_external_initializer_info`,
`value_info__get_initializer_value`, `value_info__get_value_consumers`,
`value_info__get_value_num_consumers`, `value_info__get_value_producer`,
`value_info__is_constant_initializer`, `value_info__is_from_outer_scope`,
`value_info__is_graph_output`, `value_info__is_optional_graph_input`,
`value_info__is_required_graph_input`.

**EP introspection — wrapped 0.3.0 (15):** `ep_assigned_node__get_domain`, `ep_assigned_node__get_name`,
`ep_assigned_node__get_operator_type`, `ep_assigned_subgraph__get_ep_name`,
`ep_assigned_subgraph__get_nodes`, `ep_device__device`, `ep_device__ep_metadata`,
`ep_device__ep_options`, `ep_device__memory_info`, `hardware_device__device_id`,
`hardware_device__metadata`, `hardware_device__type`, `hardware_device__vendor`,
`hardware_device__vendor_id`, `session__get_ep_graph_assignment_info`.

**Device-EP incompatibility internals (5):** `device_ep_incompatibility_details__get_error_code`,
`device_ep_incompatibility_details__get_notes`, `device_ep_incompatibility_details__get_reasons_bitmask`,
`get_hardware_device_ep_incompatibility_details`, `release_device_ep_incompatibility_details`.

**Sub-API gateways (2):** `get_training_api`, `get_execution_provider_api`.

**Typed provider-options structs — alt path (23):** `create_cann_provider_options`,
`create_cuda_provider_options`, `create_dnnl_provider_options`, `create_rocm_provider_options`,
`create_tensor_rt_provider_options`, `update_cann_provider_options`, `update_cuda_provider_options`,
`update_dnnl_provider_options`, `update_rocm_provider_options`, `update_tensor_rt_provider_options`,
`update_tensor_rt_provider_options_with_value`, `release_cann_provider_options`,
`release_cuda_provider_options`, `release_dnnl_provider_options`, `release_rocm_provider_options`,
`release_tensor_rt_provider_options`, `get_cann_provider_options_as_string`,
`get_cuda_provider_options_as_string`, `get_dnnl_provider_options_as_string`,
`get_rocm_provider_options_as_string`, `get_tensor_rt_provider_options_as_string`,
`get_cuda_provider_options_by_name`, `get_tensor_rt_provider_options_by_name`.

**Legacy EP append — alt path (8):** `session_options_append_execution_provider`,
`session_options_append_execution_provider_cann`, `session_options_append_execution_provider_cuda`,
`session_options_append_execution_provider_cuda_v2`, `session_options_append_execution_provider__dnnl`,
`session_options_append_execution_provider_rocm`,
`session_options_append_execution_provider__tensor_rt`,
`session_options_append_execution_provider__tensor_rt_v2`.

**Genuine gaps — Tier 1 (35):** `create_lora_adapter`, `create_lora_adapter_from_array`,
`release_lora_adapter`, `run_options_add_active_lora_adapter`, `get_compatibility_info_from_model`,
`get_compatibility_info_from_model_bytes`, `get_model_compatibility_for_ep_devices`,
`add_external_initializers`, `add_external_initializers_from_files_in_memory`,
`create_external_initializer_info`, `release_external_initializer_info`,
`external_initializer_info__get_byte_size`, `external_initializer_info__get_file_offset`,
`external_initializer_info__get_file_path`, `create_sync_stream_for_ep_device`,
`run_options_set_sync_stream`, `sync_stream__get_handle`, `release_sync_stream`,
`run_options_enable_profiling`, `run_options_disable_profiling`,
`run_options_get_run_log_severity_level`, `run_options_get_run_log_verbosity_level`,
`create_env_with_custom_logger`, `create_env_with_custom_logger_and_global_thread_pools`,
`create_env_with_options`, `update_env_with_custom_log_level`, `set_user_logging_function`,
`logger__get_logging_severity_level`, `logger__log_message`, `fill_string_tensor_element`,
`get_string_tensor_element`, `get_string_tensor_element_length`,
`get_resized_string_tensor_element_buffer`, `get_hardware_devices`, `get_num_hardware_devices`.

**Genuine gaps — Tier 2 (77):** `has_value`, `is_tensor`, `get_type_info`, `get_denotation_from_type_info`,
`cast_type_info_to_tensor_info`, `cast_type_info_to_map_type_info`, `cast_type_info_to_sequence_type_info`,
`cast_type_info_to_optional_type_info`, `get_map_key_type`, `get_map_value_type`, `release_map_type_info`,
`get_sequence_element_type`, `release_sequence_type_info`, `get_optional_contained_type_info`,
`tensor_type_and_shape__has_shape`, `get_tensor_shape_element_count`,
`get_tensor_element_type_and_shape_data_reference`, `tensor_at`, `create_value`, `create_opaque_value`,
`get_opaque_value`, `get_value_info_name`, `get_value_info_type_info`, `create_key_value_pairs`,
`add_key_value_pair`, `get_key_value`, `remove_key_value_pair`, `create_and_register_allocator_v2`,
`create_shared_allocator`, `get_shared_allocator`, `release_shared_allocator`, `create_memory_info_v2`,
`compare_memory_info`, `allocator_get_info`, `register_allocator`, `unregister_allocator`,
`clone_session_options`, `set_deterministic_compute`, `set_language_projection`,
`set_optimized_model_file_path`, `set_symbolic_dimensions`, `set_ep_dynamic_options`,
`session_options_set_load_cancellation_flag`, `set_per_session_thread_pool_callbacks`,
`session_options_set_custom_create_thread_fn`, `session_options_set_custom_join_thread_fn`,
`session_options_set_custom_thread_creation_options`, `session_options_set_ep_selection_policy_delegate`,
`get_session_execution_mode`, `get_mem_pattern_enabled`, `get_session_config_entry`,
`has_session_config_entry`, `get_session_options_config_entries`, `get_run_config_entry`,
`run_options_get_run_tag`, `run_options_unset_terminate`, `session_get_overridable_initializer_count`,
`session_get_overridable_initializer_name`, `session_get_overridable_initializer_type_info`,
`get_available_providers`, `release_available_providers`, `get_build_info_string`,
`get_current_gpu_device_id`, `set_current_gpu_device_id`, `enable_telemetry_events`,
`disable_telemetry_events`, `enable_ort_custom_ops`, `register_custom_ops_library`,
`register_custom_ops_library_v2`, `register_custom_ops_using_function`,
`kernel_info_get_attribute_array_string`, `kernel_info__get_config_entries`,
`register_execution_provider_library`, `unregister_execution_provider_library`, `get_bound_output_names`,
`create_tensor_with_data_and_deleter_as_ort_value`, `get_error_code`, `get_error_message`.

*(211 total = 46 + 15 + 5 + 2 + 23 + 8 + 35 + 77. `get_error_code`/`get_error_message` likely covered
via the `Status` handle — verify before wrapping.)*
