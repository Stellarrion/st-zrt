#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the base model + LoRA adapter for the LoRA positive test.

Base graph (`Y = MatMul(X,W) + MatMul(MatMul(X,a),b)`, `W = I4`):
  - without the adapter, `a = 0`, `b = 0`  => `Y == X`
  - with the adapter, `a`, `b` are non-zero  => `Y` diverges from `X`

`a`/`b` are model initializers named `lora_param_a`/`lora_param_b`; the ORT adapter
(exported via `onnxruntime.AdapterFormat`) overrides those initializers by name at run
time when the adapter is attached to the `RunOptions`.

Regenerate after editing::

    python3 st-zrt/tests/fixtures/gen_lora.py
"""

from __future__ import annotations

import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(__file__)

W = np.eye(4, dtype=np.float32)
A0 = np.zeros((4, 1), dtype=np.float32)
B0 = np.zeros((1, 4), dtype=np.float32)

nodes = [
    helper.make_node("MatMul", ["X", "W"], ["base"]),
    helper.make_node("MatMul", ["X", "lora_param_a"], ["xa"]),
    helper.make_node("MatMul", ["xa", "lora_param_b"], ["delta"]),
    helper.make_node("Add", ["base", "delta"], ["Y"]),
]
graph = helper.make_graph(
    nodes,
    "lora_base",
    [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 4]),
        # `lora_param_a`/`lora_param_b` are OVERRIDABLE INITIALIZERS: declared as graph inputs AND
        # given default (zero) initializers, so ORT's lora adapter can override them by name at run
        # time. Without the input declaration ORT rejects the override ("Invalid input name").
        helper.make_tensor_value_info("lora_param_a", TensorProto.FLOAT, [4, 1]),
        helper.make_tensor_value_info("lora_param_b", TensorProto.FLOAT, [1, 4]),
    ],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 4])],
    [
        numpy_helper.from_array(W, "W"),
        numpy_helper.from_array(A0, "lora_param_a"),
        numpy_helper.from_array(B0, "lora_param_b"),
    ],
)
model = helper.make_model(
    graph,
    producer_name="st-zrt",
    opset_imports=[helper.make_opsetid("", 21)],
)
model.ir_version = 10
onnx.save(model, os.path.join(HERE, "lora_base.onnx"))

adapter = ort.AdapterFormat()
adapter.set_adapter_version(1)
adapter.set_model_version(1)
adapter.set_parameters(
    {
        "lora_param_a": ort.OrtValue.ortvalue_from_numpy(
            np.array([[1.0], [1.0], [1.0], [1.0]], dtype=np.float32)
        ),
        "lora_param_b": ort.OrtValue.ortvalue_from_numpy(
            np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        ),
    }
)
adapter.export_adapter(os.path.join(HERE, "lora_adapter.onnx_adapter"))
print("generated lora_base.onnx + lora_adapter.onnx_adapter")
