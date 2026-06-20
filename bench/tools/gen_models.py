#!/usr/bin/env python3
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Generate controlled-size ONNX models for the st-zrt benchmarks/tests.

These are synthetic relay models (``Y = X + C``, C a constant initializer of 1.0) with a
single float tensor input ``X [1, n]`` and an identically-shaped float output ``Y``. They
let us measure the tensor-size-dependent costs the MNIST model cannot (3 KB in / 40 B out):

  - **Large INPUT**  - the O3 copy-vs-zero-copy crossover (RESULTS.md sec.3, ~5 MB).
  - **Large OUTPUT** - the E2 IoBinding zero-copy-output win (RESULTS.md sec.7).

``Add(X, C)`` (two DISTINCT inputs: the feed ``X`` and the constant ``C``) is used because it
is the realistic op (a biased add) and keeps the output genuinely materialized into a distinct
buffer each run (what E2 measures). An earlier note claimed ``Add(X, X)`` self-add segfaults in
ORT's ``Run``; that does NOT reproduce under the current code (100k iters clean, Y = 6.0
verified) — it was a symptom of the since-fixed OrtEnv use-after-free in the bench harness, not
an ORT edge case.

Usage:  python3 gen_models.py <out_dir> [<n_elements> ...]
        # default sizes: 64K (256 KB), 1M (4 MB), 4M (16 MB) of f32
"""

from __future__ import annotations

import os
import sys

import onnx
from onnx import TensorProto, helper

DEFAULT_SIZES = [65536, 1_048_576, 4_194_304]  # 256 KB, 4 MB, 16 MB of f32


def make_relay(n: int, out_path: str) -> None:
    """Emit relay_<n>.onnx: Y = X + C, X/Y shape [1, n] float32, C a constant of 1.0."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, n])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, n])
    # Constant second input (initializer) so the two Add inputs are distinct tensors.
    c = helper.make_tensor("C", TensorProto.FLOAT, [1, n], [1.0] * n, raw=False)
    add = helper.make_node("Add", ["X", "C"], ["Y"])
    graph = helper.make_graph([add], "relay", [x], [y], initializer=[c])
    model = helper.make_model(
        graph,
        producer_name="st-zrt-bench",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, out_path)


def make_relay_external(n: int, out_dir: str, label: str) -> None:
    """Emit relay_external_<label>.onnx plus relay_external_<label>.bin for C."""
    path = os.path.join(out_dir, f"relay_external_{label}.onnx")
    location = f"relay_external_{label}.bin"
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, n])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, n])
    c = helper.make_tensor("C", TensorProto.FLOAT, [1, n], [1.0] * n, raw=False)
    add = helper.make_node("Add", ["X", "C"], ["Y"])
    graph = helper.make_graph([add], "relay_external", [x], [y], initializer=[c])
    model = helper.make_model(
        graph,
        producer_name="st-zrt-bench",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=location,
        size_threshold=0,
        convert_attribute=False,
    )


def human(n: int) -> str:
    return {65536: "256k", 1_048_576: "4m", 4_194_304: "16m"}.get(n, str(n))


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    sizes = [int(a) for a in sys.argv[2:]] or DEFAULT_SIZES
    os.makedirs(out_dir, exist_ok=True)
    for n in sizes:
        label = human(n)
        path = os.path.join(out_dir, f"relay_{label}.onnx")
        make_relay(n, path)
        make_relay_external(n, out_dir, label)


if __name__ == "__main__":
    main()
