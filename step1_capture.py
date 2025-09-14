import cv2
import numpy as np
import onnxruntime
import os
import time
import sys

# Helper function to get the correct path for data files (like the ONNX model)
# whether running as a script or as a frozen .exe.
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # not in PyInstaller bundle
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- Model Loading ---
# Define the path to the ONNX model files
model_path = resource_path("model.onnx")

ort_session = None
print("--- ChromaKey AI Starting ---")
try:
    print("Attempting to load model with QNNExecutionProvider (NPU)...")
    ort_session = onnxruntime.InferenceSession(model_path, providers=['QNNExecutionProvider', 'CPUExecutionProvider'])
    print(f"Providers selected: {ort_session.get_providers()}")
    if 'QNNExecutionProvider' in ort_session.get_providers():
        print("SUCCESS: NPU provider is active.")
    else:
        print("NOTE: NPU provider was not used. Fell back to CPU.")
except Exception as e:
    print(f"NPU initialization failed: {e}")
    print("Attempting to load with CPUExecutionProvider only...")
    try:
        ort_session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print(f"Providers selected: {ort_session.get_providers()}")
        print("SUCCESS: Model loaded on CPU.")
    except Exception as fallback_e:
        print(f"FATAL: Failed to load model on CPU as well: {fallback_e}")
        ort_session = None

if not ort_session:
    print("Could not initialize ONNX Runtime session. Exiting.")
    sys.exit(1)


def main():
    print(f"[{time.time()}] Starting main function.")

    """
    Task 1: Environment setup and basic video stream.
    Goal: Ensure the development environment is ready and can successfully capture and display a real-time camera feed.
    """
    print(f"[{time.time()}] Attempting to open camera.")
    # Use OpenCV to capture the default camera (0 is usually the default).
    cap = cv2.VideoCapture(0)
    print(f"[{time.time()}] Camera capture object created.")

    # Check if the camera opened successfully.
    if not cap.isOpened():
        print(f"[{time.time()}] Error: Could not open video device. Please check if the camera is connected or used by another program.")
        return

    print(f"[{time.time()}] Camera opened successfully. Press 'q' to quit the window.")

    save_dir = "output_masks"
    os.makedirs(save_dir, exist_ok=True)  # Create folder if it doesn't exist
    frame_count = 0  # Add this before your while loop

    # Loop to continuously read frames from the camera.
    while True:
        print(f"[{time.time()}] Inside video loop: Attempting to read frame.")
        # ret is a boolean indicating if a frame was read successfully. frame is the captured image.
        ret, frame = cap.read()
        print(f"[{time.time()}] Inside video loop: Frame read attempt finished. ret={ret}")

        # If ret is False, it means no frame was captured (camera disconnected or end of video file).
        if not ret:
            print(f"[{time.time()}] Error: Can't receive frame. Exiting ...")
            break

        # Preprocess the frame for the AI model
        # Get model input shape dynamically
        input_shape = ort_session.get_inputs()[0].shape
        model_height = input_shape[2]
        model_width = input_shape[3]

        # 1. Resize to model input size
        resized_frame = cv2.resize(frame, (model_width, model_height))

        # 2. Normalize pixel values from [0, 255] to [0, 1]
        normalized_frame = resized_frame.astype('float32') / 255.0

        # 3. Adjust data dimensions (HWC to CHW) and add batch dimension (1, C, H, W)
        # OpenCV reads as BGR, so we keep it as is for now, assuming model expects BGR or handles it.
        # If the model expects RGB, an additional cv2.cvtColor(normalized_frame, cv2.COLOR_BGR2RGB) would be needed.
        # For now, let's assume the model is fine with BGR or we'll adjust later.
        input_tensor = normalized_frame.transpose(2, 0, 1)  # HWC to CHW
        input_tensor = np.expand_dims(input_tensor, axis=0) # Add batch dimension (1, C, H, W)

        # For verification, you can print the shape and type of the preprocessed tensor
        print(f"[{time.time()}] Preprocessed tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

        # Perform inference if the model was loaded successfully
        if ort_session:
            print(f"[{time.time()}] Performing inference.")
            # Get the model's input name
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name

            # Run inference
            # The input needs to be a dictionary where keys are input names and values are the input tensors
            outputs = ort_session.run([output_name], {input_name: input_tensor})
            mask = outputs[0] # Assuming the first output is the mask

            # TODO: Further processing of the mask (e.g., visualize, apply to original frame)
            # For now, we just print its properties.
            print(f"[{time.time()}] Inference complete. Mask obtained.")

            # 人像分割和背景虚化
            mask_img = np.squeeze(mask)
            if mask_img.ndim == 3:
                mask_img = mask_img[0]
            # 归一化到0~1
            mask_img = np.clip(mask_img, 0, 1)
            mask_resized = cv2.resize(mask_img, (frame.shape[1], frame.shape[0]))
            mask_3c = np.repeat(mask_resized[:, :, np.newaxis], 3, axis=2)

            # 反转掩码：人像为1，背景为0
            mask_3c = (mask_3c > 0.5).astype(np.float32)
            mask_3c = 1 - mask_3c

            # 背景虚化（加大模糊程度，比如用更大的核）
            blurred = cv2.GaussianBlur(frame, (51, 51), 0)  # 原来是(31, 31)，现在更大
            result = (frame * mask_3c + blurred * (1 - mask_3c)).astype(np.uint8)

            # 显示分割结果
            cv2.imshow('Portrait Segmentation & Background Blur', result)
            # 不保存分割结果到文件夹

        print(f"[{time.time()}] Attempting to display frame.")
        # Display the raw, unprocessed video frame in a window.
        # 'ChromaKey AI - Step 1: Camera Capture' is the window title.
        cv2.imshow('ChromaKey AI - Step 1: Camera Capture', frame)
        print(f"[{time.time()}] Frame displayed.")

        # Wait for 1ms and check for a key press. If 'q' is pressed, exit the loop.
        # The & 0xFF is a bitmask to ensure correct key ASCII value on different systems.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"[{time.time()}] 'q' pressed. Exiting loop.")
            break


    # After the loop, release the camera resource.
    cap.release()
    # Close all windows created by OpenCV.
    cv2.destroyAllWindows()
    print("Program closed and resources released.")

if __name__ == '__main__':
    main()
