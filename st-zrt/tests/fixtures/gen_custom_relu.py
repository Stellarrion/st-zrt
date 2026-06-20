#!/usr/bin/env python3
# Copyright (c) ONNX Project Contributors
# (fixture generator for st-zrt; see header — this is a test asset, not shipped code)
# SPDX-License-Identifier: Apache-2.0
"""Generate a minimal ONNX model that calls a custom operator.

The graph is `y = MyRelu(x)`, where ``MyRelu`` lives in the ``com.example``
domain (opset 1). st-zrt registers that domain — carrying a ``MyRelu``
``custom_op!`` vtable — on the session, so ORT resolves the unknown op to the
Rust kernel and invokes its ``compute``. This is the fixture for
``tests/custom_op_run.rs`` and the proof that the custom-op surface runs
end-to-end (create/compute/destroy actually fire, not just compile).

Regenerate after editing::

    python3 st-zrt/tests/fixtures/gen_custom_relu.py

The in/out shape is fixed ([4]) so the op needs *no* shape inference to load:
the model's output value_info carries the shape. (st-zrt's optional
``infer_shapes`` hook is exercised separately.)
"""

from __future__ import annotations

import os

import onnx
from onnx import TensorProto, helper

DOMAIN = "com.example"
SHAPE = [4]  # fixed in/out shape — static value_info, no shape inference needed

node = helper.make_node("MyRelu", ["x"], ["y"], name="relu_node", domain=DOMAIN)
x = helper.make_tensor_value_info("x", TensorProto.FLOAT, SHAPE)
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, SHAPE)
graph = helper.make_graph([node], "custom_relu_graph", [x], [y])

model = helper.make_model(
    graph,
    producer_name="st-zrt",
    opset_imports=[
        helper.make_opsetid("", 21),  # base ai.onnx opset (conventionally required)
        helper.make_opsetid(DOMAIN, 1),  # the custom domain our op lives in
    ],
)
# Down-pin the IR version so ORT 1.26 accepts the model regardless of the
# onnx-Python default (onnx 1.20 emits a newer IR version).
model.ir_version = 10

out = os.path.join(os.path.dirname(__file__), "custom_relu.onnx")
onnx.save_model(model, out)

# Variant with an UNSHAPED output (FLOAT element type, no dims) — ORT must call the op's
# InferOutputShape to learn the output shape, so this is the fixture for the shape-inference
# test. Without a working infer_shapes the session would fail to build (unknown output shape).
y_unshaped = onnx.ValueInfoProto()
y_unshaped.name = "y"
y_unshaped.type.tensor_type.elem_type = TensorProto.FLOAT  # no .shape → unknown
graph_u = helper.make_graph([node], "custom_relu_unshaped_graph", [x], [y_unshaped])
model_u = helper.make_model(
    graph_u,
    producer_name="st-zrt",
    opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid(DOMAIN, 1)],
)
model_u.ir_version = 10
out_u = os.path.join(os.path.dirname(__file__), "custom_relu_unshaped.onnx")
onnx.save_model(model_u, out_u)
