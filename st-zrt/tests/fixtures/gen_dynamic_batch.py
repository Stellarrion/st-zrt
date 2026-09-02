#!/usr/bin/env python3
# Copyright (c) ONNX Project Contributors
# (fixture generator for st-zrt; see header — this is a test asset, not shipped code)
# SPDX-License-Identifier: Apache-2.0
"""Generate a tiny dynamic-batch ONNX model for the CUDA-graph multi-batch test.

The graph is ``Y = Gemm(Relu(Gemm(X, W1, B1)), W2, B2)`` with ``X`` shaped
``[batch, 32]`` where ``batch`` is a SYMBOLIC dimension, so the same model runs at
batch=1 and batch=2 — two distinct input shapes that st-zrt's ``DynamicIoRuntime``
captures as two separate CUDA graphs (one ``gpu_graph_id`` per shape bucket) when built
with ``enable_cuda_graph=true``.

A pure GEMM+Relu graph is fully CUDA-graph-capturable (no host synchronization), so this
is the reliable fixture for ``tests/cuda_ep.rs``'s cuda-graph capture/replay/release test.

Regenerate after editing::

    python3 st-zrt/tests/fixtures/gen_dynamic_batch.py
"""

from __future__ import annotations

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

BATCH = "batch"  # symbolic batch dimension
IN_DIM = 32
HID = 16
OUT_DIM = 4

# Deterministic weights/biases: small positive values so relu does not zero everything.
w1 = numpy_helper.from_array(np.full((IN_DIM, HID), 0.125, np.float32), name="W1")
b1 = numpy_helper.from_array(np.full((HID,), 0.5, np.float32), name="B1")
w2 = numpy_helper.from_array(np.full((HID, OUT_DIM), 0.25, np.float32), name="W2")
b2 = numpy_helper.from_array(np.full((OUT_DIM,), 1.0, np.float32), name="B2")

gemm1 = helper.make_node(
    "Gemm", ["X", "W1", "B1"], ["H"], name="gemm1"
)  # H = X·W1 + B1
relu = helper.make_node("Relu", ["H"], ["R"], name="relu")
gemm2 = helper.make_node(
    "Gemm", ["R", "W2", "B2"], ["Y"], name="gemm2"
)  # Y = R·W2 + B2

x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [BATCH, IN_DIM])
y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [BATCH, OUT_DIM])

graph = helper.make_graph(
    [gemm1, relu, gemm2], "dynamic_batch_graph", [x], [y], [w1, b1, w2, b2]
)

model = helper.make_model(
    graph,
    producer_name="st-zrt",
    opset_imports=[helper.make_opsetid("", 21)],
)
# Down-pin the IR version so the checked-in fixture stays loadable across the ORT versions
# ZRT supports, regardless of the onnx-Python default.
model.ir_version = 10

out = os.path.join(os.path.dirname(__file__), "dynamic_batch.onnx")
onnx.save_model(model, out)
print(f"wrote {out} (input X: [{BATCH}, {IN_DIM}] -> output Y: [{BATCH}, {OUT_DIM}])")
