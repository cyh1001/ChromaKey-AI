import cv2
import numpy as np
import onnxruntime
import mediapipe as mp
import os

# Load background images
backgrounds = [
    cv2.imread('bg1.jpg'),
    cv2.imread('bg2.jpg'),
    cv2.imread('bg3.jpg')
]
bg_index = 0

# Load ONNX segmentation model
MODEL_PATH = r"./sinet-sinet-float.onnx/model.onnx/model.onnx"
ort_session = onnxruntime.InferenceSession(MODEL_PATH, providers=['QNNExecutionProvider','CPUExecutionProvider'])

def segment_frame(frame):
    input_shape = ort_session.get_inputs()[0].shape
    model_height = input_shape[2]
    model_width = input_shape[3]
    resized_frame = cv2.resize(frame, (model_width, model_height))
    normalized_frame = resized_frame.astype('float32') / 255.0
    input_tensor = normalized_frame.transpose(2, 0, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    outputs = ort_session.run([output_name], {input_name: input_tensor})
    mask = outputs[0]
    mask_img = np.squeeze(mask)
    if mask_img.ndim == 3:
        mask_img = mask_img[0]
    mask_img = np.clip(mask_img, 0, 1)
    mask_resized = cv2.resize(mask_img, (frame.shape[1], frame.shape[0]))
    mask_3c = np.repeat(mask_resized[:, :, np.newaxis], 3, axis=2)
    mask_3c = (mask_3c > 0.5).astype(np.float32)
    return mask_3c

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def classify_gesture(landmarks):
    # Example: If tip of index finger is above PIP joint, it's "Open"
    if landmarks[8].y < landmarks[6].y:
        return "Open Hand"
    else:
        return "Closed Hand"

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    last_gesture = "No Hand"
    gesture_cooldown = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = "No Hand"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = classify_gesture(hand_landmarks.landmark)
                # Switch background on open hand, with cooldown to avoid rapid switching
                if gesture == "Open Hand" and last_gesture != "Open Hand" and gesture_cooldown == 0:
                    bg_index = (bg_index + 1) % len(backgrounds)
                    print(f"Switched to background {bg_index + 1}")
                    gesture_cooldown = 20  # frames to wait before next switch
                last_gesture = gesture
        else:
            last_gesture = "No Hand"

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