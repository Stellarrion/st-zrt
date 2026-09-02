#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the fixture for the custom-op `Op::invoke` test (feature `custom-ops`).

The graph is `y = InvokeAdd(x)` in the `com.example` domain (opset 1). The Rust
kernel (`InvokeAdd` in `tests/custom_op_run.rs`) does NOT compute the add itself:
it creates the *builtin* ONNX `Add` op via `Op::create` and runs it via
`Op::invoke` (so `create`/`compute` fire the `CreateOp`/`InvokeOp` FFI bridges).
With input `[1,2,3,4]` and an internal addend of ones, the expected output is
`[2,3,4,5]`.

Regenerate after editing::

    python3 st-zrt/tests/fixtures/gen_invoke_add.py
"""

from __future__ import annotations

import os

import onnx
from onnx import TensorProto, helper

DOMAIN = "com.example"
SHAPE = [4]

node = helper.make_node(
    "InvokeAdd", ["x"], ["y"], name="invoke_add_node", domain=DOMAIN
)
x = helper.make_tensor_value_info("x", TensorProto.FLOAT, SHAPE)
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, SHAPE)
graph = helper.make_graph([node], "invoke_add_graph", [x], [y])

model = helper.make_model(
    graph,
    producer_name="st-zrt",
    opset_imports=[
        helper.make_opsetid("", 21),
        helper.make_opsetid(DOMAIN, 1),
    ],
)
model.ir_version = 10

out = os.path.join(os.path.dirname(__file__), "invoke_add.onnx")
onnx.save_model(model, out)
