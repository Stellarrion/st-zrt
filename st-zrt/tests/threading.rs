use st_zrt::{
    Environment, GraphOptimizationLevel, LoggingLevel, Session, SessionOptions, ThreadManager,
    ThreadingOptions,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};

fn upsample_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("ort_compat")
        .join("upsample.onnx")
}

#[derive(Default)]
struct ThreadStats {
    created: AtomicUsize,
    joined: AtomicUsize,
    active: AtomicUsize,
}

struct CountingThread {
    stats: Arc<ThreadStats>,
    join: JoinHandle<()>,
}

struct CountingThreadManager {
    stats: Arc<ThreadStats>,
}

impl ThreadManager for CountingThreadManager {
    type Thread = CountingThread;

    fn create(&self, work: impl FnOnce() + Send + 'static) -> st_zrt::Result<Self::Thread> {
        let stats = self.stats.clone();
        stats.created.fetch_add(1, Ordering::AcqRel);
        stats.active.fetch_add(1, Ordering::AcqRel);
        let join = thread::spawn(work);
        Ok(CountingThread { stats, join })
    }

    fn join(thread: Self::Thread) -> st_zrt::Result<()> {
        thread
            .join
            .join()
            .map_err(|_| st_zrt::Error::local("custom ORT worker panicked"))?;
        thread.stats.joined.fetch_add(1, Ordering::AcqRel);
        thread.stats.active.fetch_sub(1, Ordering::AcqRel);
        Ok(())
    }
}

#[test]
fn global_thread_pool_uses_custom_thread_manager_and_joins_workers() {
    let path = upsample_path();
    assert!(path.exists(), "missing fixture {}", path.display());

    let stats = Arc::new(ThreadStats::default());
    {
        let threading = ThreadingOptions::new()
            .expect("threading options")
            .with_intra_threads(2)
            .expect("intra threads")
            .with_inter_threads(2)
            .expect("inter threads")
            .disable_spinning()
            .expect("spin control")
            .with_thread_manager(CountingThreadManager {
                stats: stats.clone(),
            })
            .expect("thread manager");
        let env = Environment::new_with_global_thread_pools(
            LoggingLevel::Warning,
            "zrt-global-thread-pool-test",
            threading,
        )
        .expect("env");
        let sess = Session::new(
            &env,
            path.to_str().unwrap(),
            SessionOptions::new()
                .with_opt_level(GraphOptimizationLevel::Basic)
                .use_global_thread_pool(),
        )
        .expect("session");
        assert_eq!(sess.input_count(), 1);
        assert!(
            stats.created.load(Ordering::Acquire) > 0,
            "ORT did not create any custom-managed worker threads"
        );
    }

    assert_eq!(
        stats.active.load(Ordering::Acquire),
        0,
        "all custom ORT workers should be joined after env drop"
    );
    assert_eq!(
        stats.joined.load(Ordering::Acquire),
        stats.created.load(Ordering::Acquire),
        "every custom ORT worker should be joined"
    );
}
