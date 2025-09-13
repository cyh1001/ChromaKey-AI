import cv2
import numpy as np
import onnxruntime
import os
import time

# Path to the MediaPipe Face Detector ONNX model
model_path = r"C:\Users\qc_de\Desktop\edgeai\ChromaKey-AI\mediapipe_face-facedetector-float.onnx\model.onnx\model.onnx"

try:
    ort_session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    print(f"ONNX model loaded successfully from: {model_path}")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    ort_session = None

def preprocess(frame, input_shape):
    model_height = input_shape[2]
    model_width = input_shape[3]
    resized = cv2.resize(frame, (model_width, model_height))
    normalized = resized.astype('float32') / 255.0
    input_tensor = normalized.transpose(2, 0, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    print("Camera opened successfully. Press 'q' to quit.")

    save_dir = "output_faces"
    os.makedirs(save_dir, exist_ok=True)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        input_shape = ort_session.get_inputs()[0].shape
        input_tensor = preprocess(frame, input_shape)

        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: input_tensor})

        # MediaPipe Face Detector output: [1, num_detections, 16]
        # Each detection: [score, xmin, ymin, xmax, ymax, ...]
        detections = outputs[0]
        for det in detections[0]:
            score = det[0]
            if score > 0.5:  # Confidence threshold
                xmin = int(det[1] * frame.shape[1])
                ymin = int(det[2] * frame.shape[0])
                xmax = int(det[3] * frame.shape[1])
                ymax = int(det[4] * frame.shape[0])
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, f"{score:.2f}", (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Save the frame with detections
        save_path = os.path.join(save_dir, f"face_{frame_count:05d}.png")
        cv2.imwrite(save_path, frame)
        frame_count += 1

        cv2.imshow('MediaPipe Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program closed and resources released.")

if __name__ == '__main__':
    main()