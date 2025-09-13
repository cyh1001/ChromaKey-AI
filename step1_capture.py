import cv2

def main():
    """
    Task 1: Environment setup and basic video stream.
    Goal: Ensure the development environment is ready and can successfully capture and display a real-time camera feed.
    """
    # Use OpenCV to capture the default camera (0 is usually the default).
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully.
    if not cap.isOpened():
        print("Error: Could not open video device. Please check if the camera is connected or used by another program.")
        return

    print("Camera opened successfully. Press 'q' to quit the window.")

    # Loop to continuously read frames from the camera.
    while True:
        # ret is a boolean indicating if a frame was read successfully. frame is the captured image.
        ret, frame = cap.read()

        # If ret is False, it means no frame was captured (camera disconnected or end of video file).
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        # Display the raw, unprocessed video frame in a window.
        # 'ChromaKey AI - Step 1: Camera Capture' is the window title.
        cv2.imshow('ChromaKey AI - Step 1: Camera Capture', frame)

        # Wait for 1ms and check for a key press. If 'q' is pressed, exit the loop.
        # The & 0xFF is a bitmask to ensure correct key ASCII value on different systems.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop, release the camera resource.
    cap.release()
    # Close all windows created by OpenCV.
    cv2.destroyAllWindows()
    print("Program closed and resources released.")

if __name__ == '__main__':
    main()
