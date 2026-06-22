//! st-zrt-sys codegen — emits `st-zrt-sys/src/generated/api.rs` from the ORT C header.
//!
//! NOT bindgen: we run `gcc -E -P` to fully expand `struct OrtApi` (bindgen can't — the
//! function-pointer table is macro-defined), then parse the now-regular fields and map
//! C types to Rust in the zrt namespace (no `Ort*` names). Run at dev time; the output
//! is checked in (version-mirrored to libonnxruntime).
//!
//!   cargo run -p st-zrt-sys-codegen -- \<header\> \<out.rs\>
//!   cargo run -p st-zrt-sys-codegen -- \<header\> \<out.rs\> --preprocessed \<pp.c\>
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::process::{Command, Stdio};

// ─── model ───────────────────────────────────────────────────────────────────
#[derive(Debug, Clone)]
struct Param {
    rust_type: String,
    name: String,
}
#[derive(Debug, Clone)]
struct Field {
    idx: usize,
    name: String,
    ret: String, // Rust return type
    params: Vec<Param>,
    c_sig: String, // original C signature line, for the doc comment
}

fn main() {
    let mut args = std::env::args_os().skip(1);
    let header: PathBuf = args
        .next()
        .expect("usage: codegen <header> <out.rs> [--preprocessed <pp.c>]")
        .into();
    let out: PathBuf = args
        .next()
        .expect("usage: codegen <header> <out.rs>")
        .into();
    let mut preprocessed: Option<PathBuf> = None;
    let mut a = args;
    while let Some(flag) = a.next() {
        if flag == "--preprocessed" {
            preprocessed = Some(a.next().expect("--preprocessed needs a path").into());
        }
    }

    let pp = match preprocessed {
        Some(p) => std::fs::read_to_string(&p).expect("read preprocessed"),
        None => preprocess(&header),
    };
    let fields = parse_api_struct(&pp, "OrtApi");
    if fields.is_empty() {
        panic!("codegen: no OrtApi fields parsed — header layout changed?");
    }
    // The deref-style sub-API structs (each returned by a gateway getter on OrtApi). Their
    // full definitions live in the preprocessed header; emit them as typed repr(C) structs
    // so the gateway getters return `*const <Name>Api` (a real function table, not opaque).
    let sub_apis: Vec<(&str, &str)> = vec![
        ("OrtModelEditorApi", "ModelEditorApi"),
        ("OrtCompileApi", "CompileApi"),
        ("OrtInteropApi", "InteropApi"),
        ("OrtEpApi", "EpApi"),
    ];
    let sub_apis: Vec<(&str, &str, Vec<Field>)> = sub_apis
        .into_iter()
        .map(|(c, r)| {
            let fs = parse_api_struct(&pp, c);
            if fs.is_empty() {
                eprintln!("codegen: WARN `{c}` parsed 0 fields");
            }
            (c, r, fs)
        })
        .collect();
    let src = emit(&fields, &sub_apis);
    std::fs::write(&out, src).expect("write out.rs");
    report(&fields);
    eprintln!(
        "codegen: wrote {} ({} OrtApi fields + {} sub-APIs)",
        out.display(),
        fields.len(),
        sub_apis.len()
    );
}

fn preprocess(header: &std::path::Path) -> String {
    let out = Command::new("gcc")
        .args(["-E", "-P"])
        .arg(header)
        .stderr(Stdio::inherit())
        .output()
        .expect("run gcc -E -P (is gcc installed?)");
    assert!(out.status.success(), "gcc -E -P failed");
    String::from_utf8(out.stdout).expect("preprocessed output is utf-8")
}

/// Extract a `struct <name> { … };` block and parse each field line. Works for the
/// positional `OrtApi` AND the deref-style sub-APIs (`OrtModelEditorApi`, …) — the field
/// syntax is identical (`RETTYPE (* NAME)(PARAMS)`); only the EMIT differs.
fn parse_api_struct(pp: &str, struct_name: &str) -> Vec<Field> {
    let needle = format!("struct {struct_name} {{");
    let start = pp
        .find(&needle)
        .unwrap_or_else(|| panic!("codegen: `{needle}` not found"))
        + needle.len();
    let rest = &pp[start..];
    let end = rest
        .find("\n};")
        .unwrap_or_else(|| panic!("codegen: `{struct_name}` close `}};` not found"));
    let body = &rest[..end];

    let mut fields = Vec::new();
    // A field declaration ends with ';'. Most are one line, but a few (e.g.
    // GetKeyValuePairs) span multiple lines — join until ';' so indices stay correct.
    let mut buf = String::new();
    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        buf.push_str(line);
        buf.push(' ');
        if line.ends_with(';') {
            if let Some(mut f) = parse_field(buf.trim()) {
                f.idx = fields.len(); // 0-based position within this struct
                fields.push(f);
            }
            buf.clear();
        }
    }
    fields
}

/// Parse `RETTYPE(* NAME)(PARAMS) __attribute__(...);` → Field.
/// Handles both `(* NAME)` and `( * NAME)` spacing (the `Release*` family uses the latter).
fn parse_field(line: &str) -> Option<Field> {
    let line = line.trim();
    let c_sig = line.trim_end_matches(';').to_string();
    // The first '(' begins the declarator `(* NAME)` (RETTYPE has no parens).
    let open = line.find('(')?;
    let ret_c = line[..open].trim();
    let mut inner = &line[open + 1..];
    inner = inner.trim_start();
    inner = inner.strip_prefix('*')?;
    // NAME runs to the next ')'.
    let close = inner.find(')')?;
    let name = inner[..close].trim().to_string();
    // After the close paren: the param list '(...)'.
    let after = &inner[close + 1..];
    let popen = after.find('(')?;
    let params_rest = &after[popen + 1..];
    // Depth-aware: a param list can contain a function-pointer type like
    // `void (*fn)(void*, size_t)`, whose ')' must not be mistaken for the list's close.
    let pclose = match_close_paren(params_rest)?;
    let params_c = &params_rest[..pclose];

    Some(Field {
        idx: 0, // assigned by caller via position
        name,
        ret: map_return_type(ret_c),
        params: parse_params(params_c),
        c_sig,
    })
}

fn parse_params(c: &str) -> Vec<Param> {
    let c = c.trim();
    if c.is_empty() || c == "void" {
        return Vec::new();
    }
    // Split on top-level commas only — a callback like `void (*fn)(void*, size_t)`
    // contains commas inside its own parens that must not split the param.
    split_top_level(c)
        .into_iter()
        .enumerate()
        .map(|(i, p)| parse_param(&p, i))
        .collect()
}

fn parse_param(c: &str, idx: usize) -> Param {
    let c = c.trim();
    // Function-pointer parameter (a callback), e.g. `void (*fn)(void*, size_t)` — map
    // to a typed `Option<unsafe extern "C" fn(..)>` before the general pointer path.
    if let Some(p) = parse_fn_ptr_param(c) {
        return p;
    }
    // C array declarators (`x[]`, `x[N]`) decay to pointers — strip trailing `[...]`
    // chunks from the declarator and fold them into the type as pointer levels.
    let (base, array_levels) = strip_array_suffix(c);
    let (ty_c, name) = match split_param_name(&base) {
        Some((t, n)) => (t, n),
        None => (base.as_str(), format!("_arg{idx}")),
    };
    let ty_full = if array_levels > 0 {
        format!("{ty_c} {}", "*".repeat(array_levels))
    } else {
        ty_c.to_string()
    };
    Param {
        rust_type: map_type(&ty_full),
        name: sanitize_ident(&name),
    }
}

/// Remove trailing `[...]` array declarators; return the remainder + how many levels.
fn strip_array_suffix(c: &str) -> (String, usize) {
    let mut s = c.trim().to_string();
    let mut levels = 0;
    loop {
        let open = match s.rfind('[') {
            Some(o) if s[o..].ends_with(']') => o,
            _ => break,
        };
        s.truncate(open);
        let trimmed = s.trim_end();
        s = trimmed.to_string();
        levels += 1;
    }
    (s, levels)
}

/// If the param ends in a bare identifier, return (type, name).
fn split_param_name(c: &str) -> Option<(&str, String)> {
    let last_space = c.rfind(' ')?;
    let (head, tail) = c.split_at(last_space + 1);
    let tail = tail.trim();
    // Name must be a bare C identifier (no `*`, `const`, `enum`, `void`).
    if tail.is_empty() || tail.contains('*') || matches!(tail, "const" | "enum" | "struct" | "void")
    {
        return None;
    }
    Some((head.trim(), tail.to_string()))
}

fn sanitize_ident(s: &str) -> String {
    // Rust keywords / reserved used as C param names.
    if matches!(
        s,
        "type" | "in" | "out" | "ref" | "match" | "move" | "as" | "fn" | "where" | "self"
    ) {
        format!("{s}_")
    } else {
        s.to_string()
    }
}

/// Index of the ')' that closes the first top-level group in `s` (depth-aware). Used
/// for the param list itself, which may contain a function-pointer type whose inner
/// ')' must not be mistaken for the list close.
fn match_close_paren(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                if depth == 0 {
                    return Some(i);
                }
                depth -= 1;
            },
            _ => {},
        }
    }
    None
}

/// Split on top-level commas only (commas nested inside parens belong to a callback's
/// own param list, e.g. `void (*fn)(void*, size_t)`).
fn split_top_level(c: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for ch in c.chars() {
        match ch {
            '(' => {
                depth += 1;
                cur.push(ch);
            },
            ')' => {
                depth -= 1;
                cur.push(ch);
            },
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
            },
            other => cur.push(other),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Parse a function-pointer parameter `RET (*name)(INNER)` → a typed callback
/// `Option<unsafe extern "C" fn(<mapped INNER types>)>`. Returns None unless `c` is a
/// named function-pointer declarator (so ordinary pointer params are unaffected).
fn parse_fn_ptr_param(c: &str) -> Option<Param> {
    let open = c.find("(*")?;
    let after_star = &c[open + 2..]; // "name)(INNER) ..."
    let close = after_star.find(')')?;
    let name = after_star[..close].trim();
    if name.is_empty() || name.contains('*') {
        return None;
    }
    let after = &after_star[close + 1..];
    let inner_open = after.find('(')?;
    let inner_rest = &after[inner_open + 1..];
    let inner_close = match_close_paren(inner_rest)?;
    let inner = &inner_rest[..inner_close];
    let inner_types: Vec<String> = if inner.trim().is_empty() || inner.trim() == "void" {
        Vec::new()
    } else {
        split_top_level(inner)
            .into_iter()
            .map(|p| map_type(&param_type_only(&p)))
            .collect()
    };
    Some(Param {
        rust_type: format!("Option<unsafe extern \"C\" fn({})>", inner_types.join(", ")),
        name: sanitize_ident(name),
    })
}

/// Type part of an (inner) param: drop a trailing bare identifier name if present.
fn param_type_only(c: &str) -> String {
    match split_param_name(c) {
        Some((t, _)) => t.trim().to_string(),
        None => c.trim().to_string(),
    }
}

// ─── C → Rust type mapping ───────────────────────────────────────────────────
/// Map a C type (possibly with pointers/const) to a Rust FFI type string.
fn map_type(c: &str) -> String {
    let toks = tokenize(c);
    let (base, base_const, depth) = analyze(&toks);
    build_rust(&base, base_const, depth)
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Const,
    Struct,
    Enum,
    Star,
    Ident(String),
}

fn tokenize(c: &str) -> Vec<Tok> {
    let spaced = c.replace('*', " * ");
    let mut out = Vec::new();
    for w in spaced.split_whitespace() {
        match w {
            "const" => out.push(Tok::Const),
            "struct" => out.push(Tok::Struct),
            "enum" => out.push(Tok::Enum),
            "*" => out.push(Tok::Star),
            other => out.push(Tok::Ident(other.to_string())),
        }
    }
    out
}

/// Returns (base_ident, base_const, pointer_depth).
fn analyze(toks: &[Tok]) -> (String, bool, usize) {
    let mut base_const = false;
    let mut i = 0;
    while i < toks.len() {
        match &toks[i] {
            Tok::Const => {
                base_const = true;
                i += 1;
            },
            Tok::Struct | Tok::Enum => {
                i += 1;
            },
            Tok::Star => break,
            Tok::Ident(_) => break,
        }
    }
    let base = match &toks.get(i) {
        Some(Tok::Ident(s)) => s.clone(),
        _ => "void".to_string(),
    };
    let depth = toks[i..].iter().filter(|t| **t == Tok::Star).count();
    (base, base_const, depth)
}

fn build_rust(base: &str, base_const: bool, depth: usize) -> String {
    let mapped = map_base(base);
    if depth == 0 {
        return mapped;
    }
    // const base (pointee is const) → all pointer levels *const so a Vec<*const T>::as_ptr()
    // matches directly; non-const base → all *mut (Vec<*mut T>::as_mut_ptr()). ABI-identical
    // either way; const is cosmetic at the FFI boundary.
    let ptr = if base_const { "*const " } else { "*mut " };
    let mut s = mapped;
    for _ in 0..depth {
        s = format!("{ptr}{s}");
    }
    s
}

fn map_base(b: &str) -> String {
    match b {
        "void" => "core::ffi::c_void".into(),
        "char" => "core::ffi::c_char".into(),
        "int" => "core::ffi::c_int".into(),
        "short" => "core::ffi::c_short".into(),
        "long" => "core::ffi::c_long".into(),
        "unsigned" => "core::ffi::c_uint".into(),
        "int8_t" => "i8".into(),
        "int16_t" => "i16".into(),
        "int32_t" => "i32".into(),
        "int64_t" => "i64".into(),
        "uint8_t" => "u8".into(),
        "uint16_t" => "u16".into(),
        "uint32_t" => "u32".into(),
        "uint64_t" => "u64".into(),
        "size_t" => "usize".into(),
        "ssize_t" => "isize".into(),
        "intptr_t" => "isize".into(),
        "uintptr_t" => "usize".into(),
        "float" => "f32".into(),
        "double" => "f64".into(),
        "bool" => "bool".into(),
        // ORT status: the typedef (no star) maps to StatusPtr; the bare type (with a
        // star, e.g. OrtStatus*) maps to the handle so the pointer level is added once.
        "OrtStatusPtr" => "StatusPtr".into(),
        "OrtStatus" => "StatusHandle".into(),
        "OrtErrorCode" => "core::ffi::c_int".into(),
        "OrtOpAttrType" => "OpAttrType".into(),
        // Sub-API vtables (returned by the gateway getters) → typed repr(C) structs
        // emitted by `emit_sub_api`, NOT opaque handles.
        "OrtModelEditorApi" => "ModelEditorApi".into(),
        "OrtCompileApi" => "CompileApi".into(),
        "OrtInteropApi" => "InteropApi".into(),
        "OrtEpApi" => "EpApi".into(),
        // enums we model in lib.rs
        "OrtAllocatorType" => "AllocatorType".into(),
        "OrtMemType" => "MemType".into(),
        "OrtLoggingLevel" => "LoggingLevel".into(),
        "OrtExecutionProviderDevicePolicy" => "ExecutionProviderDevicePolicy".into(),
        "GraphOptimizationLevel" => "GraphOptimizationLevel".into(),
        "ONNXTensorElementDataType" => "ElementType".into(),
        "ONNXType" => "OnnxType".into(),
        "OrtSparseFormat" => "SparseFormat".into(),
        "OrtSparseIndicesFormat" => "SparseIndicesFormat".into(),
        "ExecutionMode" => "ExecutionMode".into(),
        "OrtLanguageProjection" => "i32".into(),
        "OrtCompiledModelCompatibility"
        | "OrtCompileApiFlags"
        | "OrtCudnnConvAlgoSearch"
        | "OrtCustomOpInputOutputCharacteristic"
        | "OrtDeviceEpIncompatibilityReason"
        | "OrtDeviceMemoryType"
        | "OrtExternalMemoryHandleType"
        | "OrtExternalSemaphoreType"
        | "OrtGraphicsApi"
        | "OrtHardwareDeviceType"
        | "OrtMemoryInfoDeviceType" => "i32".into(),
        // callbacks (hand-mapped typedefs in lib.rs)
        "OrtLoggingFunction" => "LoggingFunction".into(),
        "OrtThreadWorkerFn" => "ThreadWorkerFn".into(),
        "OrtCustomThreadHandle" => "CustomThreadHandle".into(),
        "OrtCustomCreateThreadFn" => "CustomCreateThreadFnHandle".into(),
        "OrtCustomJoinThreadFn" => "CustomJoinThreadFnHandle".into(),
        // RunAsync callback: a top-level fn-pointer typedef used as a by-value param. Map it
        // to the real fn-pointer type so `RunAsync` is directly callable (not opaque c_void).
        "RunAsyncCallbackFn" => "Option<unsafe extern \"C\" fn(*mut core::ffi::c_void, *mut *mut ValueHandle, usize, StatusPtr)>".into(),
        // EP device selection callback: a top-level fn-pointer typedef used as a by-value
        // param. `selected` is an inout caller-provided array of `const OrtEpDevice*`.
        "EpSelectionDelegate" => "Option<unsafe extern \"C\" fn(*const *const EpDeviceHandle, usize, *const KeyValuePairsHandle, *const KeyValuePairsHandle, *mut *const EpDeviceHandle, usize, *mut usize, *mut core::ffi::c_void) -> StatusPtr>".into(),
        // ORT handle types: strip "Ort", append "Handle".
        other if other.starts_with("Ort") => format!("{}Handle", &other[3..]),
        // Fallback: opaque.
        other => {
            eprintln!("codegen: WARN unmapped base type `{other}` → c_void");
            "core::ffi::c_void".into()
        }
    }
}

/// Return types are simpler (no name). Handle void→() and const-return-pointer.
fn map_return_type(c: &str) -> String {
    let c = c.trim();
    if c == "void" {
        return "()".into();
    }
    map_type(c)
}

// ─── feature classification ──────────────────────────────────────────────────
fn classify(name: &str) -> Option<&'static str> {
    let n = name.to_ascii_lowercase();
    // Execution-provider options (GPU/accelerators) — not CPU.
    let ep = [
        "cuda", "cudnn", "tensorrt", "rocm", "openvino", "migraphx", "dnnl", "cann", "coreml",
        "armnn", "acl", "apex", "xnnpack", "nnapi", "rknpu", "tvm", "vitisai", "qnn", "directml",
    ];
    if ep.iter().any(|e| n.contains(e)) {
        return Some("ep");
    }
    // Training API.
    if [
        "train",
        "gradient",
        "optimizer",
        "loss",
        "checkpoint",
        "loradapter",
    ]
    .iter()
    .any(|k| n.contains(k))
        || n == "gettrainingapi"
    {
        return Some("training");
    }
    // Sub-API getters to gated domains.
    if matches!(
        n.as_str(),
        "getepapi" | "getmodeleditorapi" | "getcompileapi" | "getinteropapi"
    ) {
        return Some("model-editor");
    }
    // Custom-op kernel authoring (writing kernels) — gated. Library LOADING stays core.
    if n.starts_with("kernelcontext_")
        || n.starts_with("kernelinfo")
        || n == "createop"
        || n == "addattribute"
        || n == "releaseop"
        || n == "createkernelinfo"
        || n == "releasekernelinfo"
        || n.starts_with("customopdomain")
    {
        return Some("custom-ops");
    }
    None
}

// ─── emit ────────────────────────────────────────────────────────────────────
fn emit(fields: &[Field], sub_apis: &[(&str, &str, Vec<Field>)]) -> String {
    let mut s = String::new();
    s.push_str("//! GENERATED by `st-zrt-sys-codegen` from onnxruntime_c_api.h — DO NOT EDIT.\n");
    s.push_str("//! The full OrtApi function-pointer table: IDX_* indices, typed fn aliases,\n");
    s.push_str(
        "//! and Api accessors, plus the opaque handle types. zrt names; no Ort*; no bindgen.\n",
    );
    s.push_str("#![allow(non_camel_case_types, non_snake_case, dead_code, clippy::all)]\n");
    s.push_str("// Bring in the hand-written core (Api, the enums, StatusPtr, the opaque_handle! macro).\n");
    s.push_str("use super::*;\n\n");

    // Render every OrtApi field to a snippet, grouped by feature.
    let mut groups: BTreeMap<Option<&str>, Vec<&Field>> = BTreeMap::new();
    for f in fields {
        groups.entry(classify(&f.name)).or_default().push(f);
    }
    let mut snippets: Vec<String> = Vec::new();
    for fs in groups.values() {
        for f in fs {
            snippets.push(emit_field(f, classify(&f.name)));
        }
    }
    // Sub-API struct snippets — used both for opaque-handle collection and for emission.
    let sub_snippets: Vec<String> = sub_apis
        .iter()
        .map(|(_, rust, fs)| emit_sub_api(rust, fs))
        .collect();
    snippets.extend(sub_snippets.iter().cloned());

    // Collect every distinct opaque handle type the snippets reference, and declare them.
    let handles = collect_handles(&snippets);
    s.push_str("// ── opaque handle types (one per ORT opaque struct) ──────────────────────\n");
    for h in &handles {
        s.push_str(&format!("opaque_handle!({h});\n"));
    }
    s.push_str(
        "\n/// ORT worker loop passed to a custom thread manager.\n\
         pub type ThreadWorkerFn = unsafe extern \"C\" fn(ort_worker_fn_param: *mut core::ffi::c_void);\n\n\
         #[repr(C)]\n\
         pub struct CustomThreadHandleOpaque(crate::private::Opaque);\n\n\
         /// Opaque thread handle returned by a custom thread creation callback and later passed to the\n\
         /// matching join callback.\n\
         pub type CustomThreadHandle = *const CustomThreadHandleOpaque;\n\n\
         /// ORT custom thread creation callback.\n\
         pub type CustomCreateThreadFnHandle = Option<\n\
             unsafe extern \"C\" fn(\n\
                 ort_custom_thread_creation_options: *mut core::ffi::c_void,\n\
                 ort_thread_worker_fn: ThreadWorkerFn,\n\
                 ort_worker_fn_param: *mut core::ffi::c_void,\n\
             ) -> CustomThreadHandle,\n\
         >;\n\n\
         /// ORT custom thread join callback.\n\
         pub type CustomJoinThreadFnHandle =\n\
             Option<unsafe extern \"C\" fn(ort_custom_thread_handle: CustomThreadHandle)>;\n\n\
         /// `OrtStatus*` returned by ORT — null means success. Pointee is StatusHandle.\n",
    );
    s.push_str("pub type StatusPtr = *mut StatusHandle;\n\n");

    // Core first, then gated groups.
    if let Some(core) = groups.remove(&None) {
        s.push_str("// ── core (always available) ─────────────────────────────────────────────\n");
        for f in core {
            s.push_str(&emit_field(f, None));
        }
    }
    let feature_label = |f: &str| -> String {
        match f {
            "ep" => "Execution-provider options (GPU/accelerators)".into(),
            "training" => "Training API".into(),
            "custom-ops" => "Custom-op kernel authoring".into(),
            "model-editor" => "Sub-API getters / graph editing".into(),
            other => other.into(),
        }
    };
    for (feat, fs) in groups {
        s.push_str(&format!(
            "\n// ── {} ────────────────────────────────────────────────────────────────\n",
            feature_label(feat.unwrap_or("core"))
        ));
        for f in fs {
            s.push_str(&emit_field(f, feat));
        }
    }

    // Sub-API structs (model-editor feature): typed #[repr(C)] function tables returned
    // by the gateway getters. Accessed by DEREFERENCE (`(*ptr).field`), not positional
    // index — so field order mirrors the C declaration (ABI-critical for repr(C)).
    s.push_str(
        "\n// ── Sub-API structs (deref-style function tables; model-editor feature) ────\n",
    );
    for snip in &sub_snippets {
        s.push_str(snip);
    }
    s
}

/// Emit a sub-API as a typed `#[repr(C)]` struct whose fields are `Option<fn>` (the
/// deref-style function table). Field order mirrors the C declaration order.
fn emit_sub_api(rust_name: &str, fields: &[Field]) -> String {
    let mut s =
        format!("#[cfg(feature = \"model-editor\")]\n#[repr(C)]\npub struct {rust_name} {{\n");
    for f in fields {
        let params = f
            .params
            .iter()
            .map(|p| format!("{}: {}", p.name, p.rust_type))
            .collect::<Vec<_>>()
            .join(", ");
        s.push_str(&format!(
            "    pub {}: Option<unsafe extern \"C\" fn({params}) -> {}>,\n",
            sanitize_ident(&f.name),
            f.ret
        ));
    }
    s.push_str("}\n");
    s
}

/// Collect distinct `<Name>Handle` identifiers referenced across the snippets.
fn collect_handles(snippets: &[String]) -> Vec<String> {
    let mut set: BTreeSet<String> = BTreeSet::new();
    for snip in snippets {
        for tok in snip.split(|c: char| !c.is_alphanumeric() && c != '_') {
            if tok.ends_with("Handle") {
                if matches!(
                    tok,
                    "CustomThreadHandle"
                        | "CustomCreateThreadFnHandle"
                        | "CustomJoinThreadFnHandle"
                ) {
                    continue;
                }
                set.insert(tok.to_string());
            }
        }
    }
    set.into_iter().collect()
}

fn emit_field(f: &Field, feature: Option<&str>) -> String {
    let screaming = to_screaming_snake(&f.name);
    let pascal = to_pascal(&f.name);
    let snake = to_snake(&f.name);
    let params = f
        .params
        .iter()
        .map(|p| format!("{}: {}", p.name, p.rust_type))
        .collect::<Vec<_>>()
        .join(", ");
    let cfg = match feature {
        Some(feat) => format!("#[cfg(feature = \"{feat}\")]\n"),
        None => String::new(),
    };
    format!(
        "// {c_sig}\n{cfg}pub const IDX_{screaming}: usize = {idx};\n\
         {cfg}pub type {pascal}Fn = unsafe extern \"C\" fn({params}) -> {ret};\n\
         {cfg}impl Api {{\n  #[inline]\n  {cfg}pub unsafe fn {snake}(&self) -> {pascal}Fn {{ unsafe {{ self.f(IDX_{screaming}) }} }}\n}}\n\n",
        c_sig = f.c_sig,
        cfg = cfg,
        screaming = screaming,
        idx = f.idx,
        pascal = pascal,
        params = params,
        ret = f.ret,
        snake = snake,
    )
}

// ─── name transforms ─────────────────────────────────────────────────────────
fn to_screaming_snake(name: &str) -> String {
    split_words(name)
        .map(|w| w.to_ascii_uppercase())
        .collect::<Vec<_>>()
        .join("_")
}
fn to_pascal(name: &str) -> String {
    split_words(name)
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_ascii_uppercase().to_string() + c.as_str(),
                None => String::new(),
            }
        })
        .collect::<String>()
}
fn to_snake(name: &str) -> String {
    split_words(name)
        .map(|w| w.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join("_")
}

/// Split a CamelCase / mixed identifier into lowercase word fragments.
fn split_words(name: &str) -> impl Iterator<Item = String> + '_ {
    let mut cur = String::new();
    let mut out: Vec<String> = Vec::new();
    let chars: Vec<char> = name.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_uppercase() && !cur.is_empty() {
            // Boundary if previous was lowercase, or next is lowercase (e.g. "IOFile" → IO, File).
            let prev_lower = cur
                .chars()
                .last()
                .map(|p| p.is_ascii_lowercase())
                .unwrap_or(false);
            let next_lower = chars
                .get(i + 1)
                .map(|n| n.is_ascii_lowercase())
                .unwrap_or(false);
            if prev_lower || next_lower {
                out.push(std::mem::take(&mut cur));
            }
        }
        cur.push(c);
        i += 1;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out.into_iter().map(|w| w.to_ascii_lowercase())
}

// ─── coverage report ─────────────────────────────────────────────────────────
fn report(fields: &[Field]) {
    let mut core = 0usize;
    let mut gated: BTreeMap<&str, usize> = BTreeMap::new();
    for f in fields {
        match classify(&f.name) {
            None => core += 1,
            Some(g) => *gated.entry(g).or_default() += 1,
        }
    }
    eprintln!(
        "codegen coverage: total={} core={} gated={:?}",
        fields.len(),
        core,
        gated
    );
}
