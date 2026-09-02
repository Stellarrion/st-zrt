"""Generate a model with one OVERRIDABLE initializer.

  Y = Add(X, C), where C is BOTH a graph input AND an initializer (default [2,2,2,2]).
  ORT treats such an initializer as overridable — it can be replaced at run time via a feed,
  so SessionGetOverridableInitializerCount returns 1 (C).

Used by tests/smoke.rs::session_overridable_initializers_introspect.
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper

OUT = os.path.dirname(os.path.abspath(__file__))
C = np.full(4, 2.0, dtype=np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
# C as a graph INPUT (in addition to being an initializer) makes it overridable.
Cvi = helper.make_tensor_value_info("C", TensorProto.FLOAT, [4])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
c_init = helper.make_tensor("C", TensorProto.FLOAT, [4], C.tobytes(), raw=True)
add = helper.make_node("Add", ["X", "C"], ["Y"])
graph = helper.make_graph([add], "overridable_add", [X, Cvi], [Y], initializer=[c_init])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
model.ir_version = 10
onnx.save(model, os.path.join(OUT, "overridable_add.onnx"))
print("wrote", os.path.join(OUT, "overridable_add.onnx"))
