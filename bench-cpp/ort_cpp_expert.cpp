#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace {

struct BenchCase {
  std::vector<int64_t> shape;
  std::vector<int64_t> output_shape;
  std::size_t elements;
  int default_iters;
};

BenchCase bench_case(const std::string &name) {
  if (name == "mnist") {
    return {{1, 1, 28, 28}, {1, 10}, 784, 200000};
  }
  if (name == "relay4m") {
    const std::size_t n = 1ull << 20;
    return {{1, static_cast<int64_t>(n)}, {1, static_cast<int64_t>(n)}, n, 30000};
  }
  if (name == "relay16m") {
    const std::size_t n = 1ull << 22;
    return {{1, static_cast<int64_t>(n)}, {1, static_cast<int64_t>(n)}, n, 10000};
  }
  std::cerr << "unknown case '" << name << "'; use mnist, relay4m, or relay16m\n";
  std::exit(2);
}

double mean(const std::vector<double> &xs) {
  return std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0]
              << " <model.onnx> <mnist|relay4m|relay16m> [iters] [intra_threads]\n";
    return 2;
  }

  const char *model_path = argv[1];
  const std::string case_name = argv[2];
  const BenchCase cfg = bench_case(case_name);
  const int iters = argc > 3 ? std::atoi(argv[3]) : cfg.default_iters;
  const int intra_threads = argc > 4 ? std::atoi(argv[4]) : 0;
  constexpr int repeats = 7;
  constexpr int warmup = 64;

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "zrt-cpp-expert");
  Ort::SessionOptions opts;
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (intra_threads > 0) {
    opts.SetIntraOpNumThreads(intra_threads);
  }
  Ort::Session session(env, model_path, opts);
  Ort::AllocatorWithDefaultOptions allocator;
  auto input_name = session.GetInputNameAllocated(0, allocator);
  auto output_name = session.GetOutputNameAllocated(0, allocator);

  Ort::MemoryInfo mem =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<float> input_buf(cfg.elements, 3.0f);
  std::vector<float> output_buf(static_cast<std::size_t>(
      std::accumulate(cfg.output_shape.begin(), cfg.output_shape.end(), int64_t{1},
                      std::multiplies<int64_t>())));
  Ort::Value input = Ort::Value::CreateTensor<float>(
      mem, input_buf.data(), input_buf.size(), cfg.shape.data(), cfg.shape.size());
  Ort::Value output = Ort::Value::CreateTensor<float>(
      mem, output_buf.data(), output_buf.size(), cfg.output_shape.data(),
      cfg.output_shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput(input_name.get(), input);
  binding.BindOutput(output_name.get(), output);
  Ort::RunOptions run_options;

  volatile float sink = 0.0f;
  for (int i = 0; i < warmup; ++i) {
    session.Run(run_options, binding);
    sink += output_buf[0];
  }

  std::vector<double> samples;
  samples.reserve(repeats);
  for (int r = 0; r < repeats; ++r) {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      session.Run(run_options, binding);
      sink += output_buf[0];
    }
    const auto end = std::chrono::steady_clock::now();
    const auto ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    samples.push_back(static_cast<double>(ns) / 1000.0 / static_cast<double>(iters));
  }

  std::sort(samples.begin(), samples.end());
  std::cout << "case=" << case_name << " iters=" << iters
            << " intra_threads=" << intra_threads << " sink=" << sink << "\n";
  std::cout << "ort_cpp_expert best_us=" << samples.front()
            << " median_us=" << samples[samples.size() / 2]
            << " mean_us=" << mean(samples) << "\n";
  return 0;
}
