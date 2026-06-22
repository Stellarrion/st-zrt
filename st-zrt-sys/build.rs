//! st-zrt-sys build script — from-scratch libonnxruntime integration (no bindgen, no
//! system tooling).
//!
//! Pipeline (pure Rust, no shell-outs): pin a version that mirrors libonnxruntime →
//! fetch the official onnxruntime C/C++ release package over HTTPS (ureq + rustls) →
//! SHA-256 verify (sha2) → extract (flate2+tar for `.tgz`) → link +
//! set rpath so the dylib is found at runtime. The FFI types themselves are hand-written
//! in src/.
//!
//! Supported CPU targets (all SHA-256 pinned, supply-chain verified):
//!   linux-x64, linux-aarch64, osx-arm64
//! (ORT 1.27.0 ships no osx-x86_64 build, and no win-x64 CPU archive on GitHub releases —
//! Windows-x64 CPU users must supply ST_ZRT_ORT_PATH or the NuGet Microsoft.ML.OnnxRuntime.)
//!
//! GPU (feature `cuda`): downloads the GPU libonnxruntime (linux-x64-gpu_cuda13) and rpaths a
//! system CUDA 13.x toolkit. ORT 1.27 deprecated the CUDA 12 packages and ships a CUDA 13 GPU
//! build; nvidia-*-cu13 wheels are not yet published on PyPI, so the CUDA 13 runtime libs are
//! expected on the host. They are resolved from ST_ZRT_CUDA13_PATH → CUDA_PATH → /opt/cuda
//! (default), and cuDNN 9 (`libcudnn.so.9`) must also be present on the system.
//!
//! Override: set `ST_ZRT_ORT_PATH=/path/to/onnxruntime` (an already-extracted dir with
//! `include/` and `lib/`) to skip downloading entirely.
//!
//! NOTE on target detection: build.rs is compiled for the *host*, so `#[cfg(target_*)]`
//! here reflects the host, not the cross-compile target. We therefore branch on the
//! `TARGET` triple at runtime (`env::var("TARGET")`), which is correct under
//! cross-compilation; the only `#[cfg]` is `#[cfg(unix)]` to gate `std::os::unix` APIs
//! that must compile only on a unix host.
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::io::{self, Read as _};
use std::path::{Path, PathBuf};

/// Mirrors libonnxruntime exactly. Bumping this = a new release of st-zrt-sys.
const ORT_VERSION: &str = "1.27.0";

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
            "st-zrt-sys: ORT {ORT_VERSION} publishes no win-x64 CPU archive on GitHub releases \
             (Windows CPU is now win-arm64/arm64x only). Set ST_ZRT_ORT_PATH to a pre-extracted \
             onnxruntime, or install the NuGet Microsoft.ML.OnnxRuntime package."
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

/// HTTPS download to a file (ureq + rustls; no system curl).
fn download(url: &str, dest: &Path) {
    let resp = ureq::get(url)
        .call()
        .unwrap_or_else(|e| panic!("st-zrt-sys: download {url} failed: {e}"));
    let mut file = fs::File::create(dest)
        .unwrap_or_else(|e| panic!("st-zrt-sys: create {}: {e}", dest.display()));
    let mut reader = resp.into_reader();
    io::copy(&mut reader, &mut file)
        .unwrap_or_else(|e| panic!("st-zrt-sys: write {}: {e}", dest.display()));
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
    let target = env::var("TARGET").unwrap_or_default();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let gpu = cfg!(feature = "cuda");
    println!("cargo:rerun-if-env-changed=ST_ZRT_ORT_PATH");
    println!("cargo:rerun-if-env-changed=ST_ZRT_CUDA13_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let extract_dir = match env::var("ST_ZRT_ORT_PATH") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            let (slug, ext, expected) = asset_for(&target, gpu);
            let asset = format!("onnxruntime-{slug}-{ORT_VERSION}.{ext}");
            let url = format!(
                "https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{asset}"
            );
            let marker = out_dir.join(format!("st-zrt-ort-{ORT_VERSION}-{slug}.done"));
            let extract_dir = out_dir.join("onnxruntime");

            if !marker.exists() {
                let archive = out_dir.join(&asset);
                if !archive.exists() {
                    println!("st-zrt-sys: downloading {url}");
                    download(&url, &archive);
                }
                let got = sha256_file(&archive);
                assert_eq!(
                    got, expected,
                    "st-zrt-sys: SHA-256 mismatch for {asset}\n  expected {expected}\n  got      {got}\n  supply-chain verification FAILED"
                );
                println!("st-zrt-sys: sha256 verified ({expected})");
                if extract_dir.exists() {
                    let _ = fs::remove_dir_all(&extract_dir);
                }
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

    // rpath so the ORT dylib is found without LD_LIBRARY_PATH / DYLD_LIBRARY_PATH. The
    // `-Wl,-rpath` flag is accepted by the ELF (ld) and Mach-O (ld64) linkers; MSVC's
    // link.exe rejects it (Windows resolves the DLL via PATH / colocation with the exe).
    if !target.contains("msvc") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    }

    // CUDA 13 runtime libs (feature `cuda`): resolve the system CUDA 13 toolkit lib dir and
    // rpath it so the GPU libonnxruntime's CUDA provider (`libonnxruntime_providers_cuda.so`)
    // resolves libcudart/libcublas/libcufft/libcurand/libnvrtc at runtime. cuDNN 9
    // (`libcudnn.so.9`) must also be on the host. ORT 1.27's CUDA EP is built for CUDA 13.x;
    // nvidia-*-cu13 wheels are not on PyPI yet, so the libs are expected on the system.
    #[cfg(feature = "cuda")]
    {
        let cuda13 = resolve_cuda13_lib_dir(&target);
        println!("cargo:rustc-link-search=native={}", cuda13.display());
        if !target.contains("msvc") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda13.display());
        }
    }
}
