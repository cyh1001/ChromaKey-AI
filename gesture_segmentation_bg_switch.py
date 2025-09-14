import cv2
import numpy as np
import onnxruntime
import os

# Load background images
backgrounds = [
    cv2.imread('bg1.jpg'),
    cv2.imread('bg2.jpg'),
    cv2.imread('bg3.jpg')
]
bg_index = 0

# Load ONNX segmentation model
SEGMENTATION_MODEL_PATH = r"./sinet-sinet-float.onnx/model.onnx/model.onnx"
segmentation_session = onnxruntime.InferenceSession(SEGMENTATION_MODEL_PATH, providers=['CPUExecutionProvider'])

# Load ONNX hand detection model
HAND_MODEL_PATH = r"./hand.onnx/model.onnx"
hand_session = onnxruntime.InferenceSession(HAND_MODEL_PATH, providers=['CPUExecutionProvider'])

def segment_frame(frame):
    input_shape = segmentation_session.get_inputs()[0].shape
    model_height = input_shape[2]
    model_width = input_shape[3]
    resized_frame = cv2.resize(frame, (model_width, model_height))
    normalized_frame = resized_frame.astype('float32') / 255.0
    input_tensor = normalized_frame.transpose(2, 0, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    input_name = segmentation_session.get_inputs()[0].name
    output_name = segmentation_session.get_outputs()[0].name
    outputs = segmentation_session.run([output_name], {input_name: input_tensor})
    mask = outputs[0]
    mask_img = np.squeeze(mask)
    if mask_img.ndim == 3:
        mask_img = mask_img[0]
    mask_img = np.clip(mask_img, 0, 1)
    mask_resized = cv2.resize(mask_img, (frame.shape[1], frame.shape[0]))
    mask_3c = np.repeat(mask_resized[:, :, np.newaxis], 3, axis=2)
    mask_3c = (mask_3c > 0.5).astype(np.float32)
    return mask_3c

def detect_hands(frame):
    """Detect hands using ONNX model and return gesture classification"""
    # Get model input shape
    input_shape = hand_session.get_inputs()[0].shape
    model_height = input_shape[2]
    model_width = input_shape[3]
    
    # Preprocess frame for hand detection
    resized_frame = cv2.resize(frame, (model_width, model_height))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    normalized_frame = rgb_frame.astype('float32') / 255.0
    input_tensor = normalized_frame.transpose(2, 0, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    # Run inference
    input_name = hand_session.get_inputs()[0].name
    output_name = hand_session.get_outputs()[0].name
    outputs = hand_session.run([output_name], {input_name: input_tensor})
    
    # Process outputs to determine gesture
    # This is a simplified classification - you may need to adjust based on your model's output format
    gesture = "No Hand"
    if len(outputs) > 0:
        # Assuming the model outputs hand landmarks or detection confidence
        # You may need to adjust this based on your specific model's output format
        confidence = np.max(outputs[0]) if hasattr(outputs[0], 'max') else 0
        if confidence > 0.8:  # Threshold for hand detection
            gesture = "Open Hand"  # Simplified - you can add more sophisticated gesture recognition
    
    return gesture

cap = cv2.VideoCapture(0)

last_gesture = "No Hand"
gesture_cooldown = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    # Detect hands using ONNX model
    gesture = detect_hands(frame)
    
    # Switch background on open hand, with cooldown to avoid rapid switching
    if gesture == "Open Hand" and last_gesture != "Open Hand" and gesture_cooldown == 0:
        bg_index = (bg_index + 1) % len(backgrounds)
        print(f"Switched to background {bg_index + 1}")
        gesture_cooldown = 20  # frames to wait before next switch
    last_gesture = gesture

    if gesture_cooldown > 0:
        gesture_cooldown -= 1

    # Segmentation
    mask_3c = segment_frame(frame)
    mask_3c = 1 - mask_3c  # <-- Add this line to invert the mask
    bg = cv2.resize(backgrounds[bg_index], (frame.shape[1], frame.shape[0]))
    output = (frame * mask_3c + bg * (1 - mask_3c)).astype(np.uint8)

    cv2.putText(output, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Gesture + Segmentation + BG Switch', output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()