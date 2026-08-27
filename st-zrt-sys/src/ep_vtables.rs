//! EP-author implementor vtable structs (feature `model-editor`).
//!
//! These are the `repr(C)` function-pointer tables an execution provider *implements* and hands to
//! ORT, transcribed field-for-field from `onnxruntime_ep_c_api.h` (ORT 1.27):
//!   - [`EpFactoryVTable`]  — `struct OrtEpFactory` (the factory ORT dlopens via `CreateEpFactories`)
//!   - [`EpVTable`]         — `struct OrtEp` (one EP instance's lifecycle: capability/compile/run)
//!   - [`NodeComputeInfoVTable`] — `struct OrtNodeComputeInfo` (one fused graph's compute functions)
//!
//! Unlike [`crate::generated::EpApi`] (an ORT-owned helper table returned by `GetEpApi`), these
//! structs are *produced by the EP author*. **Field order is ABI** — ORT indexes them by slot, so a
//! field added, dropped, reordered, or mistyped here silently misroutes every callback. Every field
//! is an `Option<unsafe extern "C" fn>` (niche-optimized to a nullable pointer) so an EP may leave a
//! callback NULL (`None`) when ORT documents it optional.
//!
//! On Linux `ORT_API_CALL`, the SAL annotations (`_In_`, `_Out_`, …), and `NO_EXCEPTION` all expand
//! to nothing, so `ORT_API_T(T, N, …)` is `T (*N)(…)` and `ORT_API2_STATUS(N, …)` is
//! `OrtStatus* (*N)(…)` (a nullable status return = [`crate::StatusPtr`]).
//!
//! Field names are kept verbatim from the C header (matching [`crate::generated`]) for traceability.
#![allow(non_snake_case)] // field names mirror the C `OrtEp`/`OrtEpFactory` slots verbatim

use crate::generated::StatusPtr;
use crate::*;
use core::ffi::{c_char, c_int, c_void};

// ── EP-produced impl-struct handles ─────────────────────────────────────────
// These are produced by the EP author and returned to ORT through the vtables below. Their full
// bodies are intentionally opaque here — an EP only ever hands `*mut` pointers across the boundary,
// and the safe `EpAuthor` surface builds them from trait callbacks. (Profiling events, the one
// impl-struct already wrapped read-only, live in the parent crate's `ep_authoring`.)
opaque_handle!(EpProfilerImplHandle);
opaque_handle!(DataTransferImplHandle);
opaque_handle!(ExternalResourceImporterImplHandle);
// `OrtResourceCount` — opaque resource-availability record returned from `OrtEp::GetAvailableResource`.
opaque_handle!(ResourceCountHandle);

/// `struct OrtNodeComputeInfo` — the compute functions for one graph an [`EpVTable::Compile`]
/// produced. The EP allocates and fills one per compiled graph; ORT calls `CreateState` per run,
/// then `Compute`, then `ReleaseState`.
#[repr(C)]
#[derive(Debug)]
pub struct NodeComputeInfoVTable {
    pub ort_version_supported: u32,
    /// `CreateState(this, compute_context, &compute_state)` — build an opaque per-run state.
    pub CreateState: Option<
        unsafe extern "C" fn(
            this: *mut NodeComputeInfoVTable,
            compute_context: *mut NodeComputeContextHandle,
            compute_state: *mut *mut c_void,
        ) -> StatusPtr,
    >,
    /// `Compute(this, compute_state, kernel_context)` — run the fused graph for one inference call.
    pub Compute: Option<
        unsafe extern "C" fn(
            this: *mut NodeComputeInfoVTable,
            compute_state: *mut c_void,
            kernel_context: *mut KernelContextHandle,
        ) -> StatusPtr,
    >,
    /// `ReleaseState(this, compute_state)` — free the per-run state after the final `Compute`.
    pub ReleaseState:
        Option<unsafe extern "C" fn(this: *mut NodeComputeInfoVTable, compute_state: *mut c_void)>,
}

/// `struct OrtEp` — the vtable for one execution-provider instance. Field order matches ORT 1.27
/// exactly. Most callbacks are optional (`None`); the safe `EpAuthor` trait supplies defaults so an
/// author implements only what their EP needs.
#[repr(C)]
#[derive(Debug)]
pub struct EpVTable {
    pub ort_version_supported: u32,
    pub GetName: Option<unsafe extern "C" fn(this: *const EpHandle) -> *const c_char>,
    pub GetCapability: Option<
        unsafe extern "C" fn(
            *mut EpHandle,
            *const GraphHandle,
            *mut EpGraphSupportInfoHandle,
        ) -> StatusPtr,
    >,
    pub Compile: Option<
        unsafe extern "C" fn(
            *mut EpHandle,
            graphs: *const *const GraphHandle,
            fused_nodes: *const *const NodeHandle,
            count: usize,
            node_compute_infos: *mut *mut NodeComputeInfoVTable,
            ep_context_nodes: *mut *mut NodeHandle,
        ) -> StatusPtr,
    >,
    pub ReleaseNodeComputeInfos: Option<
        unsafe extern "C" fn(
            *mut EpHandle,
            node_compute_infos: *mut *mut NodeComputeInfoVTable,
            num: usize,
        ),
    >,
    pub GetPreferredDataLayout: Option<
        unsafe extern "C" fn(*mut EpHandle, preferred_data_layout: *mut EpDataLayout) -> StatusPtr,
    >,
    pub ShouldConvertDataLayoutForOp: Option<
        unsafe extern "C" fn(
            *mut EpHandle,
            domain: *const c_char,
            op_type: *const c_char,
            target_data_layout: EpDataLayout,
            should_convert: *mut c_int,
        ) -> StatusPtr,
    >,
    pub SetDynamicOptions: Option<
        unsafe extern "C" fn(
            *mut EpHandle,
            option_keys: *const *const c_char,
            option_values: *const *const c_char,
            num_options: usize,
        ) -> StatusPtr,
    >,
    pub OnRunStart: Option<
        unsafe extern "C" fn(*mut EpHandle, run_options: *const RunOptionsHandle) -> StatusPtr,
    >,
    pub OnRunEnd: Option<
        unsafe extern "C" fn(
            *mut EpHandle,
            run_options: *const RunOptionsHandle,
            sync_stream: bool,
        ) -> StatusPtr,
    >,
    pub CreateAllocator: Option<
        unsafe extern "C" fn(
            *mut EpHandle,
            memory_info: *const MemoryInfoHandle,
            allocator: *mut *mut AllocatorHandle,
        ) -> StatusPtr,
    >,
    pub CreateSyncStreamForDevice: Option<
        unsafe extern "C" fn(
            *mut EpHandle,
            memory_device: *const MemoryDeviceHandle,
            stream: *mut *mut SyncStreamImplHandle,
        ) -> StatusPtr,
    >,
    pub GetCompiledModelCompatibilityInfo: Option<
        unsafe extern "C" fn(this: *const EpHandle, graph: *const GraphHandle) -> *const c_char,
    >,
    pub GetKernelRegistry: Option<
        unsafe extern "C" fn(
            *mut EpHandle,
            kernel_registry: *mut *const KernelRegistryHandle,
        ) -> StatusPtr,
    >,
    pub IsConcurrentRunSupported:
        Option<unsafe extern "C" fn(*mut EpHandle, is_supported: *mut bool) -> StatusPtr>,
    pub Sync: Option<unsafe extern "C" fn(*mut EpHandle) -> StatusPtr>,
    pub CreateProfiler: Option<
        unsafe extern "C" fn(*mut EpHandle, profiler: *mut *mut EpProfilerImplHandle) -> StatusPtr,
    >,
    pub IsGraphCaptureEnabled: Option<unsafe extern "C" fn(this: *const EpHandle) -> bool>,
    pub IsGraphCaptured:
        Option<unsafe extern "C" fn(this: *const EpHandle, graph_annotation_id: c_int) -> bool>,
    pub ReplayGraph:
        Option<unsafe extern "C" fn(*mut EpHandle, graph_annotation_id: c_int) -> StatusPtr>,
    pub GetGraphCaptureNodeAssignmentPolicy:
        Option<unsafe extern "C" fn(this: *const EpHandle) -> GraphCaptureNodeAssignmentPolicy>,
    pub GetAvailableResource: Option<
        unsafe extern "C" fn(
            this: *const EpHandle,
            available: *mut ResourceCountHandle,
        ) -> StatusPtr,
    >,
    pub OnSessionInitializationEnd: Option<unsafe extern "C" fn(*mut EpHandle) -> StatusPtr>,
    pub GetDefaultMemoryDevice: Option<
        unsafe extern "C" fn(
            this: *const EpHandle,
            device: *mut *const MemoryDeviceHandle,
        ) -> StatusPtr,
    >,
    pub ReleaseCapturedGraph:
        Option<unsafe extern "C" fn(*mut EpHandle, graph_annotation_id: c_int) -> StatusPtr>,
}

/// `struct OrtEpFactory` — the factory vtable ORT obtains (in-process or via the dlopened
/// `CreateEpFactories` symbol) and uses to mint [`EpVTable`] instances. Field order matches ORT 1.27.
#[repr(C)]
#[derive(Debug)]
pub struct EpFactoryVTable {
    pub ort_version_supported: u32,
    pub GetName: Option<unsafe extern "C" fn(this: *const EpFactoryHandle) -> *const c_char>,
    pub GetVendor: Option<unsafe extern "C" fn(this: *const EpFactoryHandle) -> *const c_char>,
    pub GetSupportedDevices: Option<
        unsafe extern "C" fn(
            *mut EpFactoryHandle,
            devices: *const *const HardwareDeviceHandle,
            num_devices: usize,
            ep_devices: *mut *mut EpDeviceHandle,
            max_ep_devices: usize,
            num_ep_devices: *mut usize,
        ) -> StatusPtr,
    >,
    pub CreateEp: Option<
        unsafe extern "C" fn(
            *mut EpFactoryHandle,
            devices: *const *const HardwareDeviceHandle,
            ep_metadata_pairs: *const *const KeyValuePairsHandle,
            num_devices: usize,
            session_options: *const SessionOptionsHandle,
            logger: *const LoggerHandle,
            ep: *mut *mut EpHandle,
        ) -> StatusPtr,
    >,
    /// `ReleaseEp(factory, ep)` — `ep` is an [`EpVTable`] (the `struct OrtEp` the factory minted).
    pub ReleaseEp: Option<unsafe extern "C" fn(*mut EpFactoryHandle, ep: *mut EpVTable)>,
    pub GetVendorId: Option<unsafe extern "C" fn(this: *const EpFactoryHandle) -> u32>,
    pub GetVersion: Option<unsafe extern "C" fn(this: *const EpFactoryHandle) -> *const c_char>,
    pub ValidateCompiledModelCompatibilityInfo: Option<
        unsafe extern "C" fn(
            *mut EpFactoryHandle,
            devices: *const *const HardwareDeviceHandle,
            num_devices: usize,
            compatibility_info: *const c_char,
            model_compatibility: *mut c_int, // OrtCompiledModelCompatibility → i32
        ) -> StatusPtr,
    >,
    pub CreateAllocator: Option<
        unsafe extern "C" fn(
            *mut EpFactoryHandle,
            memory_info: *const MemoryInfoHandle,
            allocator_options: *const KeyValuePairsHandle,
            allocator: *mut *mut AllocatorHandle,
        ) -> StatusPtr,
    >,
    pub ReleaseAllocator:
        Option<unsafe extern "C" fn(*mut EpFactoryHandle, allocator: *mut AllocatorHandle)>,
    pub CreateDataTransfer: Option<
        unsafe extern "C" fn(
            *mut EpFactoryHandle,
            data_transfer: *mut *mut DataTransferImplHandle,
        ) -> StatusPtr,
    >,
    pub IsStreamAware: Option<unsafe extern "C" fn(this: *const EpFactoryHandle) -> bool>,
    pub CreateSyncStreamForDevice: Option<
        unsafe extern "C" fn(
            *mut EpFactoryHandle,
            memory_device: *const MemoryDeviceHandle,
            stream_options: *const KeyValuePairsHandle,
            stream: *mut *mut SyncStreamImplHandle,
        ) -> StatusPtr,
    >,
    pub GetHardwareDeviceIncompatibilityDetails: Option<
        unsafe extern "C" fn(
            *mut EpFactoryHandle,
            hw: *const HardwareDeviceHandle,
            details: *mut DeviceEpIncompatibilityDetailsHandle,
        ) -> StatusPtr,
    >,
    pub CreateExternalResourceImporterForDevice: Option<
        unsafe extern "C" fn(
            *mut EpFactoryHandle,
            ep_device: *const EpDeviceHandle,
            out_importer: *mut *mut ExternalResourceImporterImplHandle,
        ) -> StatusPtr,
    >,
    pub GetNumCustomOpDomains:
        Option<unsafe extern "C" fn(*mut EpFactoryHandle, num_domains: *mut usize) -> StatusPtr>,
    pub GetCustomOpDomains: Option<
        unsafe extern "C" fn(
            *mut EpFactoryHandle,
            domains: *mut *mut CustomOpDomainHandle,
            num_domains: usize,
        ) -> StatusPtr,
    >,
    pub InitGraphicsInterop: Option<
        unsafe extern "C" fn(
            *mut EpFactoryHandle,
            config: *const GraphicsInteropConfigHandle,
        ) -> StatusPtr,
    >,
    pub DeinitGraphicsInterop: Option<
        unsafe extern "C" fn(*mut EpFactoryHandle, ep_device: *const EpDeviceHandle) -> StatusPtr,
    >,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    /// A default vtable (every callback `None`) must be constructible — an EP leaves optional
    /// callbacks NULL. This also pins `repr(C)` field layout via size + offset asserts: a wrong
    /// field type, a dropped/added field, or a reorder breaks the expected sizes.
    #[test]
    fn ep_vtable_default_is_all_none_and_layout_pinned() {
        let ep = EpVTable {
            ort_version_supported: crate::API_VERSION,
            GetName: None,
            GetCapability: None,
            Compile: None,
            ReleaseNodeComputeInfos: None,
            GetPreferredDataLayout: None,
            ShouldConvertDataLayoutForOp: None,
            SetDynamicOptions: None,
            OnRunStart: None,
            OnRunEnd: None,
            CreateAllocator: None,
            CreateSyncStreamForDevice: None,
            GetCompiledModelCompatibilityInfo: None,
            GetKernelRegistry: None,
            IsConcurrentRunSupported: None,
            Sync: None,
            CreateProfiler: None,
            IsGraphCaptureEnabled: None,
            IsGraphCaptured: None,
            ReplayGraph: None,
            GetGraphCaptureNodeAssignmentPolicy: None,
            GetAvailableResource: None,
            OnSessionInitializationEnd: None,
            GetDefaultMemoryDevice: None,
            ReleaseCapturedGraph: None,
        };
        assert_eq!(ep.ort_version_supported, 27);
        assert_eq!(offset_of!(EpVTable, ort_version_supported), 0);
        // version (4) + pad (4) + 24 nullable fn pointers (8 each) = 200 on 64-bit.
        assert_eq!(size_of::<EpVTable>(), 8 + 24 * 8);
    }

    #[test]
    fn ep_factory_vtable_default_is_all_none_and_layout_pinned() {
        let f = EpFactoryVTable {
            ort_version_supported: crate::API_VERSION,
            GetName: None,
            GetVendor: None,
            GetSupportedDevices: None,
            CreateEp: None,
            ReleaseEp: None,
            GetVendorId: None,
            GetVersion: None,
            ValidateCompiledModelCompatibilityInfo: None,
            CreateAllocator: None,
            ReleaseAllocator: None,
            CreateDataTransfer: None,
            IsStreamAware: None,
            CreateSyncStreamForDevice: None,
            GetHardwareDeviceIncompatibilityDetails: None,
            CreateExternalResourceImporterForDevice: None,
            GetNumCustomOpDomains: None,
            GetCustomOpDomains: None,
            InitGraphicsInterop: None,
            DeinitGraphicsInterop: None,
        };
        assert_eq!(f.ort_version_supported, 27);
        assert_eq!(offset_of!(EpFactoryVTable, ort_version_supported), 0);
        // version (4) + pad (4) + 19 nullable fn pointers (8 each) = 160 on 64-bit.
        assert_eq!(size_of::<EpFactoryVTable>(), 8 + 19 * 8);
    }

    #[test]
    fn node_compute_info_vtable_layout_pinned() {
        // version (4) + pad (4) + 3 nullable fn pointers (8 each) = 32 on 64-bit.
        assert_eq!(size_of::<NodeComputeInfoVTable>(), 8 + 3 * 8);
        assert_eq!(offset_of!(NodeComputeInfoVTable, ort_version_supported), 0);
    }
}
