import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import onnxruntime as ort

# 1. Clean up old or split artifacts
for f in os.listdir("."):
    if f.startswith("mobilenetv3_student_diabetic_retinopathy") and (
        f.endswith(".onnx") or ".data" in f
    ):
        try:
            os.remove(f)
            print(f"Removed old artifact: {f}")
        except Exception:
            pass

# 2. Reconstruct model matching the trained continuous architecture
model = models.mobilenet_v3_small(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, 1)  # Single continuous scalar output
)

checkpoint_name = "student_mobilenet_distilled.pth"
checkpoint = torch.load(checkpoint_name, map_location="cpu")

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

# 3. Export with 384x384 resolution and dynamic batch axes
dummy_input = torch.randn(1, 3, 384, 384)
output_onnx = "mobilenetv3_student_diabetic_retinopathy.onnx"

torch.onnx.export(
    model,
    dummy_input,
    output_onnx,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input_image"],
    output_names=["severity_score"],
    dynamic_axes={
        "input_image": {0: "batch_size"},
        "severity_score": {0: "batch_size"}
    }
)

size_mb = round(os.path.getsize(output_onnx) / (1024 * 1024), 2)
print(f"Standalone export complete: {output_onnx} ({size_mb} MB)")

# 4. Immediate numerical parity check
with torch.no_grad():
    pt_val = model(dummy_input).view(-1).numpy()

session = ort.InferenceSession(output_onnx, providers=["CPUExecutionProvider"])
ort_val = session.run(None, {"input_image": dummy_input.numpy()})[0].flatten()

diff = np.max(np.abs(pt_val - ort_val))
print(f"Max absolute discrepancy (PyTorch vs ONNX): {diff:.6e}")
assert diff < 1e-4, "Parity check failed!"
print("Verification successful: ONNX artifact is ready for app.py.")

# 3. Export with legacy TorchScript engine and opset 18
dummy_input = torch.randn(1, 3, 384, 384)
output_onnx = "mobilenetv3_student_diabetic_retinopathy.onnx"

torch.onnx.export(
    model,
    dummy_input,
    output_onnx,
    export_params=True,
    opset_version=18,           # Use native modern opset supported by your torch version
    do_constant_folding=True,
    input_names=["input_image"],
    output_names=["severity_score"],
    dynamic_axes={
        "input_image": {0: "batch_size"},
        "severity_score": {0: "batch_size"}
    },
    dynamo=False                # Forces self-contained single-file export
)

size_mb = round(os.path.getsize(output_onnx) / (1024 * 1024), 2)
print(f"Standalone export complete: {output_onnx} ({size_mb} MB)")