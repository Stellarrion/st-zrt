//! st-zrt-sys build script — from-scratch libonnxruntime integration (no bindgen, no
//! system tooling).
//!
//! Pipeline (pure Rust, no shell-outs): pin a version that mirrors libonnxruntime →
//! fetch the official onnxruntime C/C++ release package over HTTPS (ureq + rustls) →
//! SHA-256 verify (sha2) → extract (flate2+tar for `.tgz`, zip for `.zip`) → link +
//! set rpath so the dylib is found at runtime. The FFI types themselves are hand-written
//! in src/.
//!
//! Supported CPU targets (all SHA-256 pinned, supply-chain verified):
//!   linux-x64, linux-aarch64, osx-arm64, win-x64
//! (ORT 1.26.0 ships no osx-x86_64 build.)
//!
//! GPU (feature `cuda`): downloads the GPU libonnxruntime (linux-x64-gpu) + the CUDA 12.x
//! runtime libs (nvidia cu12 wheels, SHA-256 pinned) and rpaths them, so `cargo build
//! --features cuda` produces a binary that runs on an NVIDIA GPU with no manual setup. ORT
//! 1.26's CUDA EP is built for CUDA 12.x (the host toolkit version is irrelevant at runtime);
//! cuDNN 9 must be on the system. Override the CUDA libs with ST_ZRT_CUDA12_PATH.
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
const ORT_VERSION: &str = "1.26.0";

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
        // GPU libonnxruntime (CUDA EP). v0.1: linux-x64 only (tested on an RTX 4090).
        if linux && x86_64 {
            return (
                "linux-x64-gpu",
                "tgz",
                "cb7df7ee2ca0f962c7ce7c839aeae36223d146a91fb4646d62fb0046f297479f",
            );
        }
        panic!(
            "st-zrt-sys: `cuda` feature needs the GPU libonnxruntime; ORT {ORT_VERSION} ships a \
             tested linux-x64-gpu build only in v0.1 (got TARGET '{target}'). Set ST_ZRT_ORT_PATH \
             to a pre-extracted GPU onnxruntime."
        );
    }
    if linux && x86_64 {
        (
            "linux-x64",
            "tgz",
            "1254da24fb389cf39dc0ff3451ab48301740ffbfcbaf646849df92f80ee92c57",
        )
    } else if linux && aarch64 {
        (
            "linux-aarch64",
            "tgz",
            "34ff1c2d0f12e2cf3d33a0c5f82e39792e1d581fbd6968fd7c30d173654be01a",
        )
    } else if darwin && aarch64 {
        (
            "osx-arm64",
            "tgz",
            "7a1280bbb1701ea514f71828765237e7896e0f2e1cd332f1f70dbd5c3e33aca3",
        )
    } else if windows && x86_64 {
        (
            "win-x64",
            "zip",
            "6ebe99b5564bf4d029b6e93eac9ff423682b6212eade769e9ca3f685eaf500b4",
        )
    } else if darwin && x86_64 {
        panic!(
            "st-zrt-sys: ORT {ORT_VERSION} ships no osx-x86_64 build (Apple Intel is unsupported by upstream). \
             Build on arm64, or set ST_ZRT_ORT_PATH to a pre-extracted onnxruntime."
        );
    } else {
        panic!(
            "st-zrt-sys: TARGET '{target}' unsupported in v0.1 (linux-x64/aarch64, osx-arm64, win-x64). \
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

/// Extract a `.zip` into `dest` (no system unzip). Skips entries that escape `dest`.
fn extract_zip(archive: &Path, dest: &Path) {
    let f = fs::File::open(archive)
        .unwrap_or_else(|e| panic!("st-zrt-sys: open {}: {e}", archive.display()));
    let mut za = zip::ZipArchive::new(f)
        .unwrap_or_else(|e| panic!("st-zrt-sys: read zip {}: {e}", archive.display()));
    for i in 0..za.len() {
        let mut entry = match za.by_index(i) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let Some(rel) = entry.enclosed_name() else {
            continue;
        };
        let out = dest.join(rel);
        if entry.is_dir() {
            fs::create_dir_all(&out)
                .unwrap_or_else(|e| panic!("st-zrt-sys: mkdir {}: {e}", out.display()));
        } else {
            if let Some(p) = out.parent() {
                let _ = fs::create_dir_all(p);
            }
            let mut o = fs::File::create(&out)
                .unwrap_or_else(|e| panic!("st-zrt-sys: create {}: {e}", out.display()));
            io::copy(&mut entry, &mut o)
                .unwrap_or_else(|e| panic!("st-zrt-sys: copy {}: {e}", out.display()));
        }
    }
}

/// CUDA 12.x runtime libs for the GPU build (feature `cuda`). The nvidia cu12 wheels are the
/// canonical, reproducible source; each is SHA-256 pinned (supply-chain parity with the ORT
/// packages). ORT 1.26's CUDA EP links libcudart/cublas/cufft/curand (+ nvjitlink); cuDNN 9 is
/// expected on the system. `(name, files.pythonhosted.org url, sha256)`.
#[cfg(feature = "cuda")]
const CUDA12_WHEELS: &[(&str, &str, &str)] = &[
    (
        "cuda-runtime",
        "https://files.pythonhosted.org/packages/bc/46/a92db19b8309581092a3add7e6fceb4c301a3fd233969856a8cbf042cd3c/nvidia_cuda_runtime_cu12-12.9.79-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "25bba2dfb01d48a9b59ca474a1ac43c6ebf7011f1b0b8cc44f54eb6ac48a96c3",
    ),
    (
        "cublas",
        "https://files.pythonhosted.org/packages/cb/c0/0a517bfe63ccd3b92eb254d264e28fca3c7cab75d07daea315250fb1bf73/nvidia_cublas_cu12-12.9.2.10-py3-none-manylinux_2_27_x86_64.whl",
        "e4f53a8ca8c5d6e8c492d0d0a3d565ecb59a751b19cfdaa4f6da0ab2104c1702",
    ),
    (
        "cufft",
        "https://files.pythonhosted.org/packages/95/f4/61e6996dd20481ee834f57a8e9dca28b1869366a135e0d42e2aa8493bdd4/nvidia_cufft_cu12-11.4.1.4-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "c67884f2a7d276b4b80eb56a79322a95df592ae5e765cf1243693365ccab4e28",
    ),
    (
        "curand",
        "https://files.pythonhosted.org/packages/31/44/193a0e171750ca9f8320626e8a1f2381e4077a65e69e2fb9708bd479e34a/nvidia_curand_cu12-10.3.10.19-py3-none-manylinux_2_27_x86_64.whl",
        "49b274db4780d421bd2ccd362e1415c13887c53c214f0d4b761752b8f9f6aa1e",
    ),
    (
        "nvjitlink",
        "https://files.pythonhosted.org/packages/46/0c/c75bbfb967457a0b7670b8ad267bfc4fffdf341c074e0a80db06c24ccfd4/nvidia_nvjitlink_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
        "e3f1171dbdc83c5932a45f0f4c99180a70de9bd2718c1ab77d14104f6d7147f9",
    ),
];

/// Download + SHA-256-verify the cu12 wheels and extract their `*.so*` into `out_dir/cuda12`.
/// Cached via a marker file. Returns the cuda12 lib dir.
#[cfg(feature = "cuda")]
fn fetch_cuda12(out_dir: &Path) -> PathBuf {
    let cuda12 = out_dir.join("cuda12");
    let marker = out_dir.join("st-zrt-cuda12.done");
    if !marker.exists() {
        if cuda12.exists() {
            let _ = fs::remove_dir_all(&cuda12);
        }
        fs::create_dir_all(&cuda12)
            .unwrap_or_else(|e| panic!("st-zrt-sys: mkdir cuda12 {}: {e}", cuda12.display()));
        for (name, url, expected) in CUDA12_WHEELS {
            let whl = out_dir.join(format!("{name}.whl"));
            if !whl.exists() {
                println!("st-zrt-sys: downloading CUDA 12 wheel {name}");
                download(url, &whl);
            }
            let got = sha256_file(&whl);
            assert_eq!(
                got, *expected,
                "st-zrt-sys: SHA-256 mismatch for CUDA wheel {name}\n  expected {expected}\n  got      {got}\n  supply-chain verification FAILED"
            );
            extract_wheel_libs(&whl, &cuda12);
        }
        let _ = fs::File::create(&marker);
    }
    cuda12
}

/// Extract `nvidia/*/lib/*.so*` from a cu12 wheel (a zip) into `dest`, flattened by basename.
#[cfg(feature = "cuda")]
fn extract_wheel_libs(archive: &Path, dest: &Path) {
    let f = fs::File::open(archive)
        .unwrap_or_else(|e| panic!("st-zrt-sys: open wheel {}: {e}", archive.display()));
    let mut za = zip::ZipArchive::new(f)
        .unwrap_or_else(|e| panic!("st-zrt-sys: read wheel {}: {e}", archive.display()));
    for i in 0..za.len() {
        let mut entry = match za.by_index(i) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let name = entry.name().to_string();
        if (name.ends_with(".so") || name.contains(".so.")) && name.contains("/lib/") {
            let Some(fname) = Path::new(&name).file_name() else {
                continue;
            };
            let out = dest.join(fname);
            let mut o = fs::File::create(&out)
                .unwrap_or_else(|e| panic!("st-zrt-sys: create {}: {e}", out.display()));
            io::copy(&mut entry, &mut o)
                .unwrap_or_else(|e| panic!("st-zrt-sys: copy {}: {e}", out.display()));
        }
    }
}

fn main() {
    let target = env::var("TARGET").unwrap_or_default();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let gpu = cfg!(feature = "cuda");
    println!("cargo:rerun-if-env-changed=ST_ZRT_ORT_PATH");
    println!("cargo:rerun-if-env-changed=ST_ZRT_CUDA12_PATH");

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
                if ext == "zip" {
                    extract_zip(&archive, &out_dir);
                } else {
                    extract_tgz(&archive, &out_dir);
                }
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

    // CUDA 12 runtime libs (feature `cuda`): download + SHA-256-verify the nvidia cu12 wheels
    // and rpath their dir so the GPU libonnxruntime's CUDA provider resolves at runtime. ORT
    // 1.26's CUDA EP is built for CUDA 12.x; the host toolkit version is irrelevant at runtime
    // — only the matching libcudart.so.12 etc. matter (cuDNN 9 stays on the system). Override
    // with ST_ZRT_CUDA12_PATH (a dir of libcudart.so.12 / libcublas.so.12 / …) to skip it.
    #[cfg(feature = "cuda")]
    {
        let cuda12 = match env::var("ST_ZRT_CUDA12_PATH") {
            Ok(p) => PathBuf::from(p),
            Err(_) => fetch_cuda12(&out_dir),
        };
        assert!(
            cuda12.is_dir(),
            "st-zrt-sys: cuda12 lib dir missing at {}",
            cuda12.display()
        );
        println!("cargo:rustc-link-search=native={}", cuda12.display());
        if !target.contains("msvc") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda12.display());
        }
    }
}
