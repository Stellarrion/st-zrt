"""Generate a model whose initializer is stored in an external data file.

  Y = Add(X, C), C = [2, 2, 2, 2] float32 externalized into a sibling .data file.

Outputs (next to this script):
  external_add.onnx        the graph (C marked external, location = DATA_NAME)
  external_add.onnx.data   the raw bytes of C

Used by tests/smoke.rs::external_initializer_files_in_memory to exercise ORT's
AddExternalInitializersFromFilesInMemory: the test copies the .onnx to a temp dir
WITHOUT the .data sibling and supplies C's bytes from memory, so session creation
succeeds only via the in-memory path.
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper

OUT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(OUT, "external_add.onnx")
DATA_NAME = "external_add.onnx.data"

C = np.full(4, 2.0, dtype=np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
c_init = helper.make_tensor("C", TensorProto.FLOAT, [4], C.tobytes(), raw=True)
add = helper.make_node("Add", ["X", "C"], ["Y"])
graph = helper.make_graph([add], "external_add", [X], [Y], initializer=[c_init])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
model.ir_version = 10

onnx.save_model(
    model,
    MODEL,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=DATA_NAME,
    size_threshold=0,  # threshold 0 => externalize every tensor (incl. C)
)
print(f"wrote {MODEL}")
print(f"wrote {os.path.join(OUT, DATA_NAME)}")
