//! st-zrt-sys build script — from-scratch libonnxruntime integration (no bindgen, no
//! system tooling).
//!
//! Pipeline (pure Rust, no shell-outs): pin a version that mirrors libonnxruntime →
//! fetch the official onnxruntime C/C++ release package over HTTPS (ureq + rustls with
//! bounded timeouts and bounded retries, streamed to `<archive>.part` and atomically
//! renamed only after the body completes) → SHA-256 verify (sha2) → extract (flate2+tar
//! for `.tgz`) → link and export the acquired lib dir. The FFI types themselves are
//! hand-written in src/.
//!
//! Supported CPU targets (all SHA-256 pinned, supply-chain verified):
//!   linux-x64, linux-aarch64, osx-arm64
//! (ORT 1.27.0 ships no osx-x86_64 build. Automatic Windows acquisition is NOT
//! implemented: upstream publishes the win-x64 CPU package as a `.zip`, and this build
//! script extracts `.tgz` archives only — see `asset_for`. Windows-x64 users must set
//! `ST_ZRT_ORT_PATH` to an already-extracted ONNX Runtime directory — from the GitHub
//! release ZIP or the NuGet `Microsoft.ML.OnnxRuntime` package — with `include/` and
//! `lib/`.)
//!
//! GPU (feature `cuda`): downloads the GPU libonnxruntime (linux-x64-gpu_cuda13; a
//! ~200 MiB archive — the download policy below is sized for it) and link-searches a
//! system CUDA 13.x toolkit. ORT 1.27 deprecated the CUDA 12 packages and ships a CUDA 13
//! GPU build; nvidia-*-cu13 wheels are not yet published on PyPI, so the CUDA 13 runtime
//! libs are expected on the host. They are resolved from ST_ZRT_CUDA13_PATH → CUDA_PATH
//! → /opt/cuda (default), and cuDNN 9 (`libcudnn.so.9`) must also be present on the
//! system (see the loader notes below).
//!
//! Override: set `ST_ZRT_ORT_PATH=/path/to/onnxruntime` (an already-extracted dir with
//! `include/` and `lib/`) to skip downloading entirely. A relative path is resolved
//! against the parent of this crate's manifest directory (the workspace root in this
//! repository), never against the current working directory.
//!
//! RUNTIME LOADING — what this script does and does not guarantee. The `-Wl,-rpath`
//! link argument emitted below reaches only linkable units of *this crate* (its own
//! tests/examples — that is what keeps `cargo test -p st-zrt-sys` working without
//! `LD_LIBRARY_PATH`). It does NOT propagate through the rlib to downstream final
//! binaries: Cargo applies `rustc-link-arg` only to the emitting package's own
//! benchmarks, binaries, cdylibs, examples, and tests, never to dependents. A consumer
//! of `st-zrt` must arrange for the dynamic loader to find `libonnxruntime` itself, e.g.:
//! - Linux: run with `LD_LIBRARY_PATH=<ort>/lib`, or have the *final binary's own* build script
//!   emit `cargo:rustc-link-arg=-Wl,-rpath,<libdir>` (see the `DEP_ONNXRUNTIME_*` metadata
//!   below), or colocate the .so with the binary plus an `$ORIGIN`-based rpath;
//! - macOS: run with `DYLD_LIBRARY_PATH=<ort>/lib`, or a consumer-side rpath;
//! - Windows: the loader searches the executable's directory and `PATH` — put `onnxruntime.dll`
//!   next to the .exe or add its directory to `PATH`.
//!
//! The CUDA build adds the same caveat for the CUDA 13 toolkit libs and cuDNN 9: the
//! link-search/rpath below covers this crate's own units only; a downstream CUDA binary needs
//! its own loader setup (`LD_LIBRARY_PATH` including the CUDA lib64 dir and the cuDNN 9 library
//! dir, or an equivalent consumer-side rpath).
//!
//! To make that possible, this script exports the acquired ORT directories to direct
//! dependents' build scripts as Cargo metadata (this crate declares `links =
//! "onnxruntime"`): `DEP_ONNXRUNTIME_ROOT` (extracted dir), `DEP_ONNXRUNTIME_LIBDIR`
//! (`<root>/lib`), and `DEP_ONNXRUNTIME_INCLUDE` (`<root>/include`). This has been
//! supported by Cargo far longer than the declared MSRV; it is a data export only —
//! nothing is loaded or linked automatically by it.
//!
//! NOTE on target detection: build.rs is compiled for the *host*, so `#[cfg(target_*)]`
//! here reflects the host, not the cross-compile target. We therefore branch on the
//! `TARGET` triple at runtime (`env::var("TARGET")`), which is correct under
//! cross-compilation; the only `#[cfg]` is `#[cfg(unix)]` to gate `std::os::unix` APIs
//! that must compile only on a unix host.
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::io::{self, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Mirrors libonnxruntime exactly. Bumping this = a new release of st-zrt-sys.
const ORT_VERSION: &str = "1.27.0";

/// Bounded TCP/TLS connect time per download attempt.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Per-read bound on the response body. Bounds a *stalled* transfer without imposing a
/// fixed budget on the whole ~200 MiB GPU archive (that is the global timeout's job).
const RECV_BODY_TIMEOUT: Duration = Duration::from_secs(120);
/// Whole-request bound per attempt: generous for the GPU archive on a slow link
/// (~200 MiB at ≈125 KiB/s), still finite.
const GLOBAL_TIMEOUT: Duration = Duration::from_secs(45 * 60);
/// Total attempts (first try + bounded retries) for one archive fetch.
const DOWNLOAD_ATTEMPTS: u32 = 3;
/// Backoff before each retry after the first failed attempt.
const RETRY_BACKOFF: Duration = Duration::from_secs(2);

/// Resolve the release asset for a target triple: `(slug, extension, pinned sha256)`.
/// Every supported target is pinned — a mismatch fails the build (supply-chain gate).
fn asset_for(target: &str, gpu: bool) -> (&'static str, &'static str, &'static str) {
    let (linux, darwin, windows, x86_64, aarch64) = (
        target.contains("linux"),
        target.contains("darwin"),
        target.contains("windows"),
        target.contains("x86_64"),
        target.contains("aarch64"),
    );
    if gpu {
        // GPU libonnxruntime (CUDA EP, CUDA 13). linux-x64 only.
        if linux && x86_64 {
            return (
                "linux-x64-gpu_cuda13",
                "tgz",
                "1a3227e1dc2f53d9f877c93278af500b15e26d99aa5ade877692138b3ab7d351",
            );
        }
        panic!(
            "st-zrt-sys: `cuda` feature needs the GPU libonnxruntime; ORT {ORT_VERSION} ships a \
             tested linux-x64-gpu_cuda13 build only (got TARGET '{target}'). Set ST_ZRT_ORT_PATH \
             to a pre-extracted GPU onnxruntime."
        );
    }
    if linux && x86_64 {
        (
            "linux-x64",
            "tgz",
            "547e40a48f1fe73e3f812d7c88a948612c23f896b91e4e2ee1e232d7b468246f",
        )
    } else if linux && aarch64 {
        (
            "linux-aarch64",
            "tgz",
            "3e4d83ac06924a32a07b6d7f91ce6f852876153fc0bbdf931bf517a140bfbe48",
        )
    } else if darwin && aarch64 {
        (
            "osx-arm64",
            "tgz",
            "545e81c58152353acb0d1e8bd6ce4b62f830c0961f5b3acfedc790ffd76e477a",
        )
    } else if windows && x86_64 {
        panic!(
            "st-zrt-sys: automatic ONNX Runtime acquisition is not implemented for Windows. \
             Upstream publishes the ORT {ORT_VERSION} win-x64 CPU package as a `.zip`, and this \
             build script extracts `.tgz` archives only. Download the win-x64 CPU ZIP from the \
             ONNX Runtime release (or the NuGet Microsoft.ML.OnnxRuntime package), extract it, \
             and set ST_ZRT_ORT_PATH to that directory (it must contain include/ and lib/)."
        );
    } else if darwin && x86_64 {
        panic!(
            "st-zrt-sys: ORT {ORT_VERSION} ships no osx-x86_64 build (Apple Intel is unsupported by upstream). \
             Build on arm64, or set ST_ZRT_ORT_PATH to a pre-extracted onnxruntime."
        );
    } else {
        panic!(
            "st-zrt-sys: TARGET '{target}' unsupported (linux-x64/aarch64, osx-arm64). \
             Set ST_ZRT_ORT_PATH to an already-extracted onnxruntime dir."
        );
    }
}

/// HTTPS agent with the bounded download policy (connect / per-read body / global
/// timeouts; rustls, no system curl). Redirects are followed by default (GitHub release
/// assets redirect to a CDN).
fn https_agent() -> ureq::Agent {
    let config = ureq::Agent::config_builder()
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_recv_body(Some(RECV_BODY_TIMEOUT))
        .timeout_global(Some(GLOBAL_TIMEOUT))
        .build();
    ureq::Agent::new_with_config(config)
}

/// Stream one response body to `part` completely; flush and fsync so the caller can
/// atomically rename it. Any mid-stream failure leaves no complete file behind.
fn download_once(url: &str, part: &Path) -> io::Result<u64> {
    let resp = https_agent()
        .get(url)
        .call()
        .map_err(|e| io::Error::other(format!("request {url} failed: {e}")))?;
    let status = resp.status();
    if status != 200 {
        return Err(io::Error::other(format!(
            "download {url} returned HTTP {status}"
        )));
    }
    let file = fs::File::create(part)
        .map_err(|e| io::Error::new(e.kind(), format!("create {}: {e}", part.display())))?;
    let mut file = io::BufWriter::with_capacity(1 << 16, file);
    // ureq 3: the body is owned by the response; split it out and stream an owned reader to disk.
    let (_, body) = resp.into_parts();
    let mut reader = body.into_reader();
    let copied = io::copy(&mut reader, &mut file)?;
    file.flush()?;
    // Durability before the atomic rename: a truncated-but-renamed archive would poison
    // the cache and only fail at SHA verification.
    file.get_ref()
        .sync_all()
        .map_err(|e| io::Error::other(format!("fsync {}: {e}", part.display())))?;
    Ok(copied)
}

/// Download `url` to `dest` atomically: bounded retries over `<dest>.part`, rename into
/// place only after one fully streamed, flushed, fsynced attempt. Stale `.part` files
/// from an interrupted earlier build are removed first and never renamed.
fn download_with_retries(url: &str, dest: &Path) {
    let part = PathBuf::from(format!("{}.part", dest.display()));
    let _ = fs::remove_file(&part);
    let mut last_error = String::new();
    for attempt in 1..=DOWNLOAD_ATTEMPTS {
        match download_once(url, &part) {
            Ok(bytes) => {
                fs::rename(&part, dest).unwrap_or_else(|e| {
                    let _ = fs::remove_file(&part);
                    panic!(
                        "st-zrt-sys: rename {} -> {} after {bytes}-byte download failed: {e}",
                        part.display(),
                        dest.display()
                    )
                });
                println!(
                    "st-zrt-sys: downloaded {url} ({bytes} bytes) -> {}",
                    dest.display()
                );
                return;
            },
            Err(e) => {
                // Never leave a partial body where a later build could mistake it for data.
                let _ = fs::remove_file(&part);
                last_error = format!("attempt {attempt}/{DOWNLOAD_ATTEMPTS}: {e}");
                println!("st-zrt-sys: download {url} failed — {last_error}");
                if attempt < DOWNLOAD_ATTEMPTS {
                    std::thread::sleep(RETRY_BACKOFF);
                }
            },
        }
    }
    panic!(
        "st-zrt-sys: download {url} failed after {DOWNLOAD_ATTEMPTS} attempts ({last_error}). \
         Check the network/proxy, or pre-download the archive to {} (or set ST_ZRT_ORT_PATH).",
        dest.display()
    );
}

/// SHA-256 of a file, lowercase hex (no system sha256sum/shasum).
fn sha256_file(path: &Path) -> String {
    let mut file =
        fs::File::open(path).unwrap_or_else(|e| panic!("st-zrt-sys: open {}: {e}", path.display()));
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file
            .read(&mut buf)
            .unwrap_or_else(|e| panic!("st-zrt-sys: read {}: {e}", path.display()));
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    hex::encode(hasher.finalize())
}

/// Put a SHA-256-verified release archive at `out_dir/<asset>`, returning the verified
/// digest. Downloads with bounded retries when missing. When a previously cached archive
/// fails verification (truncated download, bit rot, upstream re-publish), it is removed
/// and fetched once more instead of poisoning OUT_DIR forever; a fresh download that
/// still mismatches the pin fails the build here — before extraction.
fn acquire_verified_archive(url: &str, asset: &str, expected: &str, out_dir: &Path) -> String {
    let archive = out_dir.join(asset);
    if archive.exists() {
        let got = sha256_file(&archive);
        if got == expected {
            println!("st-zrt-sys: cached {asset} sha256 verified ({expected})");
            return got;
        }
        println!(
            "st-zrt-sys: cached {asset} failed SHA-256 verification\n  expected {expected}\n  got      {got}\n  removing it and fetching a fresh copy"
        );
        fs::remove_file(&archive).unwrap_or_else(|e| {
            panic!(
                "st-zrt-sys: remove corrupt archive {}: {e}",
                archive.display()
            )
        });
    }
    download_with_retries(url, &archive);
    let got = sha256_file(&archive);
    assert_eq!(
        got, expected,
        "st-zrt-sys: SHA-256 mismatch for freshly downloaded {asset}\n  expected {expected}\n  got      {got}\n  supply-chain verification FAILED — the pinned release asset changed upstream; update the pin deliberately"
    );
    got
}

/// Extract a `.tgz` (gzip tarball) into `dest` (no system tar).
fn extract_tgz(archive: &Path, dest: &Path) {
    let f = fs::File::open(archive)
        .unwrap_or_else(|e| panic!("st-zrt-sys: open {}: {e}", archive.display()));
    let gz = flate2::read::GzDecoder::new(f);
    let mut tar = tar::Archive::new(gz);
    tar.set_overwrite(true);
    tar.unpack(dest)
        .unwrap_or_else(|e| panic!("st-zrt-sys: unpack {}: {e}", archive.display()));
}

/// Resolve the system CUDA 13 toolkit lib dir for the GPU build (feature `cuda`). The CUDA EP
/// in ORT 1.27's `linux-x64-gpu_cuda13` package needs `libcudart.so.13`, `libcublas.so.13`,
/// `libcufft.so.12`, `libcurand.so.10`, `libnvrtc.so.13` (and `libcudnn.so.9` on the system).
/// `nvidia-*-cu13` wheels are not published on PyPI yet, so these libs must be present on the
/// host. Resolution order: `ST_ZRT_CUDA13_PATH` → `CUDA_PATH` → `/opt/cuda`.
///
/// Loader note: this directory is added as a link-search path and (for this crate's own
/// linkable units) an rpath. Downstream final binaries must still arrange for the loader
/// to find these libraries plus cuDNN 9 — see the runtime-loading section of the module
/// docs.
#[cfg(feature = "cuda")]
fn resolve_cuda13_lib_dir(target: &str) -> PathBuf {
    let root = env::var("ST_ZRT_CUDA13_PATH")
        .map(PathBuf::from)
        .or_else(|_| env::var("CUDA_PATH").map(PathBuf::from))
        .unwrap_or_else(|_| PathBuf::from("/opt/cuda"));
    let (libdir, probe_name) = if target.contains("windows") {
        (root.join("Bin"), "cudart64_13.dll")
    } else {
        (root.join("lib64"), "libcudart.so.13")
    };
    let probe = libdir.join(probe_name);
    assert!(
        probe.exists(),
        "st-zrt-sys: `cuda` feature needs a system CUDA 13 toolkit. \
         Looked for {} (resolved from ST_ZRT_CUDA13_PATH → CUDA_PATH → /opt/cuda, got {}). \
         nvidia-*-cu13 wheels are not on PyPI yet; install the CUDA 13.x runtime and cuDNN 9.",
        probe.display(),
        root.display()
    );
    libdir
}

fn main() {
    // Emit this unconditionally, before the DOCS_RS early return: once any
    // rerun-if-env-changed directive exists, Cargo only reruns for the listed variables,
    // so listing DOCS_RS only when it is already set would freeze the docs.rs path on.
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    // The pinned ORT version must be visible to the crate (re-exported as `ORT_VERSION`)
    // even on the DOCS_RS early-return path below, where no artifact is acquired: the pin
    // is a compile-time constant of the binding surface, not of the acquisition step.
    // Emitted from the same constant that drives the download/sha256 table.
    println!("cargo:rustc-env=ST_ZRT_ORT_VERSION={ORT_VERSION}");

    if env::var_os("DOCS_RS").is_some() {
        return;
    }

    let target = env::var("TARGET").unwrap_or_default();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let gpu = cfg!(feature = "cuda");
    println!("cargo:rerun-if-env-changed=ST_ZRT_ORT_PATH");
    println!("cargo:rerun-if-env-changed=ST_ZRT_CUDA13_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let extract_dir = match env::var("ST_ZRT_ORT_PATH") {
        Ok(p) => {
            let path = PathBuf::from(p);
            let path = if path.is_absolute() {
                path
            } else {
                // Relative paths resolve against the workspace root (the parent of this
                // crate's manifest), not against the build's working directory.
                let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
                manifest
                    .parent()
                    .expect("st-zrt-sys manifest has no workspace parent")
                    .join(path)
            }
            .canonicalize()
            .expect("st-zrt-sys: ST_ZRT_ORT_PATH does not resolve to a directory");
            assert!(
                path.is_dir(),
                "st-zrt-sys: ST_ZRT_ORT_PATH does not resolve to a directory"
            );
            path
        },
        Err(_) => {
            let (slug, ext, expected) = asset_for(&target, gpu);
            let asset = format!("onnxruntime-{slug}-{ORT_VERSION}.{ext}");
            let url = format!(
                "https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{asset}"
            );
            let marker = out_dir.join(format!("st-zrt-ort-{ORT_VERSION}-{slug}.done"));
            let extract_dir = out_dir.join("onnxruntime");

            // The marker only counts while the extraction it describes still exists; a
            // wiped/partial `onnxruntime/` dir must not keep the marker's promise alive.
            if !(marker.exists() && extract_dir.join("lib").is_dir()) {
                let _ = fs::remove_file(&marker);
                // Verified immediately before extraction — the supply-chain gate.
                acquire_verified_archive(&url, &asset, expected, &out_dir);
                if extract_dir.exists() {
                    let _ = fs::remove_dir_all(&extract_dir);
                }
                let archive = out_dir.join(&asset);
                extract_tgz(&archive, &out_dir);
                let extracted = out_dir.join(format!("onnxruntime-{slug}-{ORT_VERSION}"));
                fs::rename(&extracted, &extract_dir)
                    .expect("st-zrt-sys: rename extracted onnxruntime dir");
                let _ = fs::File::create(&marker);
            }
            extract_dir
        },
    };

    let lib = extract_dir.join("lib");
    assert!(
        lib.is_dir(),
        "st-zrt-sys: missing lib/ at {}",
        lib.display()
    );

    // Linux ships a versioned libonnxruntime.so.<ver>; ensure an unversioned symlink
    // exists so `-lonnxruntime` resolves. (Runtime target detection; the symlink API is
    // gated to a unix HOST via cfg(unix).)
    if target.contains("linux") {
        #[cfg(unix)]
        {
            let so = lib.join("libonnxruntime.so");
            if !so.exists() {
                if let Ok(entries) = fs::read_dir(&lib) {
                    let ver = entries
                        .filter_map(|e| e.ok())
                        .map(|e| e.file_name().into_string().unwrap_or_default())
                        .find(|n| n.starts_with("libonnxruntime.so."))
                        .expect("st-zrt-sys: no libonnxruntime.so.* found in lib/");
                    std::os::unix::fs::symlink(&ver, &so)
                        .expect("st-zrt-sys: create libonnxruntime.so symlink");
                }
            }
        }
    }

    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=onnxruntime");

    // Downstream-facing Cargo metadata (available to direct dependents' build scripts as
    // DEP_ONNXRUNTIME_ROOT / DEP_ONNXRUNTIME_LIBDIR / DEP_ONNXRUNTIME_INCLUDE because this
    // crate declares `links = "onnxruntime"`). A consumer's build script can use these to
    // emit its own rpath/link-arg for its FINAL binaries. They do not link or load
    // anything by themselves — see the runtime-loading section of the module docs.
    println!("cargo:root={}", extract_dir.display());
    println!("cargo:libdir={}", lib.display());
    println!("cargo:include={}", extract_dir.join("include").display());

    // rpath so THIS crate's own linkable units (e.g. `cargo test -p st-zrt-sys`) find the
    // dylib without LD_LIBRARY_PATH / DYLD_LIBRARY_PATH. The `-Wl,-rpath` flag is accepted
    // by the ELF (ld) and Mach-O (ld64) linkers; MSVC's link.exe rejects it (Windows
    // resolves the DLL via PATH / colocation with the exe). IMPORTANT: this does NOT
    // propagate to downstream final binaries — `rustc-link-arg` from a lib crate reaches
    // only that crate's own benchmarks/binaries/cdylibs/examples/tests. Downstream
    // consumers need LD_LIBRARY_PATH/DYLD_LIBRARY_PATH, DLL colocation + PATH, or their
    // own build-script rpath built on the DEP_ONNXRUNTIME_LIBDIR metadata above.
    if !target.contains("msvc") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    }

    // CUDA 13 runtime libs (feature `cuda`): resolve the system CUDA 13 toolkit lib dir
    // and link-search it (plus the same crate-local-only rpath caveat) so the GPU
    // libonnxruntime's CUDA provider (`libonnxruntime_providers_cuda.so`) can resolve
    // libcudart/libcublas/libcufft/libcurand/libnvrtc at runtime. cuDNN 9
    // (`libcudnn.so.9`) must also be on the host. ORT 1.27's CUDA EP is built for CUDA
    // 13.x; nvidia-*-cu13 wheels are not on PyPI yet, so the libs are expected on the
    // system.
    #[cfg(feature = "cuda")]
    {
        let cuda13 = resolve_cuda13_lib_dir(&target);
        println!("cargo:rustc-link-search=native={}", cuda13.display());
        if !target.contains("msvc") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda13.display());
        }
    }
}
