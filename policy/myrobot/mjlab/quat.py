import onnxruntime as ort
import numpy as np

model_path = "policy/myrobot/mjlab/policy.onnx"

sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

obs = np.zeros((1, 114), dtype=np.float32)
time_step = np.array([[0.0]], dtype=np.float32)

body_quat_w = sess.run(
    ["body_quat_w"],
    {
        "obs": obs,
        "time_step": time_step,
    },
)[0]

anchor_idx = 0
q = body_quat_w.reshape(-1)[anchor_idx * 4 : anchor_idx * 4 + 4]

print("ref_anchor_q0(wxyz) =", q)