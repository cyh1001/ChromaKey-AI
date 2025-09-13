import cv2
import numpy as np
import onnxruntime
import os
import time

# Define the path to the ONNX model
# model_path = "C:\\Users\\qc_de\\Desktop\\edgeai\\ChromaKey-AI\\model.onnx"
model_path = "models/model.onnx"
# Load the ONNX model and create an inference session
try:
    # Use CPUExecutionProvider for initial testing.
    # For Snapdragon NPU, this would eventually be ['QNNExecutionProvider', 'CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(model_path, providers=['QNNExecutionProvider', 'CPUExecutionProvider'])
    print(f"ONNX model loaded successfully from: {model_path} using providers: {ort_session.get_providers()}")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    print("Attempting to load with CPUExecutionProvider only as a fallback...")
    try:
        ort_session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print(f"ONNX model loaded successfully from: {model_path} using CPUExecutionProvider only.")
    except Exception as fallback_e:
        print(f"Error loading ONNX model with CPUExecutionProvider fallback: {fallback_e}")
        ort_session = None # Set to None if loading fails

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
        # 1. Resize to model input size (e.g., 1024x1024)
        resized_frame = cv2.resize(frame, (224, 224))

        # 2. Normalize pixel values from [0, 255] to [0, 1]
        normalized_frame = resized_frame.astype('float32') / 255.0

        # 3. Adjust data dimensions (HWC to CHW) and add batch dimension (1, C, H, W)
        # OpenCV reads as BGR, so we keep it as is for now, assuming model expects BGR or handles it.
        # If the model expects RGB, an additional cv2.cvtColor(normalized_frame, cv2.COLOR_BGR2RGB) would be needed.
        # For now, let's assume the model is fine with BGR or we'll adjust later.
        input_tensor = normalized_frame.transpose(2, 0, 1)  # HWC to CHW
        input_tensor = np.expand_dims(input_tensor, axis=0) # Add batch dimension (1, C, H, W)

        # Convert the input tensor to uint8
        input_tensor = (input_tensor * 255).astype(np.uint8)

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
