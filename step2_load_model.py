import onnxruntime
import os

def main():
    """
    Task 2: AI Model Selection and Loading
    Goal: Verify that the downloaded ONNX model can be loaded successfully by onnxruntime.
    """
    model_path = "C:\\Users\\qc_de\\Desktop\\edgeai\\ChromaKey-AI\\model.onnx"

    # 1. Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        print("Please ensure you have downloaded the model and placed it in the correct directory.")
        return

    print(f"Found model file at: '{model_path}'")
    print("Attempting to load the model with ONNX Runtime...")

    try:
        # 2. Try to load the model and create an inference session
        # When creating the session, you can specify the execution providers.
        # For Snapdragon, this would eventually be ['QNNExecutionProvider', 'CPUExecutionProvider']
        # For this initial test, we'll use the default CPU provider to keep it simple.
        ort_session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        # 3. If successful, print a success message and model info
        print("\n-------------------------------------------------")
        print("Success! Model loaded without errors.")
        print("-------------------------------------------------")

        # Get and print model input details
        input_meta = ort_session.get_inputs()
        print(f"Model Inputs: {len(input_meta)}")
        for i, input in enumerate(input_meta):
            print(f"  Input {i}:")
            print(f"    Name: {input.name}")
            print(f"    Shape: {input.shape}")
            print(f"    Type: {input.type}")

        # Get and print model output details
        output_meta = ort_session.get_outputs()
        print(f"\nModel Outputs: {len(output_meta)}")
        for i, output in enumerate(output_meta):
            print(f"  Output {i}:")
            print(f"    Name: {output.name}")
            print(f"    Shape: {output.shape}")
            print(f"    Type: {output.type}")

    except Exception as e:
        print("\n-------------------------------------------------")
        print("Error: Failed to load the ONNX model.")
        print("-------------------------------------------------")
        print(f"The following error occurred:\n{e}")
        print("\nThe model file might be corrupted, or there might be an issue with the ONNX Runtime setup.")

if __name__ == '__main__':
    main()
