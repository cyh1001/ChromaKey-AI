import onnxruntime
import sys

print(f"Python executable: {sys.executable}")
print(f"sys.path: {sys.path}")
print(f"Loaded onnxruntime from: {onnxruntime.__file__}")
print(f"Does onnxruntime have InferenceSession? {'InferenceSession' in dir(onnxruntime)}")