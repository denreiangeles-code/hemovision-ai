import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

def test_hemovision():
    print("Booting up HemoVision AI...")
    
    # 1. Load the saved brain
    try:
        model = tf.keras.models.load_model('hemovision_poc_v1.keras')
    except Exception as e:
        print("Error: Could not load the model. Did you run the training script?")
        return

    # 2. Open the Vault (The Test Set it has NEVER seen before)
    test_df = pd.read_csv('test_set.csv')
    
    # Let's grab the very first patient in the vault
    target_patient = test_df.iloc[0]
    
    print("\n--- Target Patient Loaded ---")
    print(f"Location: {target_patient['Dataset']} #{target_patient['Number']}")
    print(f"ACTUAL Hemoglobin (Ground Truth): {target_patient['Hgb']} g/dL")

    # 3. Locate the physical image safely (bypassing the Mac files)
    folder_path = os.path.join(".", target_patient['Dataset'], str(target_patient['Number']))
    img_path = None
    
    for file in os.listdir(folder_path):
        if 'palpebral' in file.lower() and not file.startswith('._'):
            img_path = os.path.join(folder_path, file)
            break

    if not img_path:
        print("Error: Could not find the image in the test folder.")
        return

    # 4. The Translation (Preparing it for the AI)
    # Open, resize, and convert to RGB
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    
    # Convert to mathematical matrix
    img_array = np.array(img)
    
    # CRITICAL: The model expects a "batch" of images, even if it's just one.
    # We must change the shape from (224, 224, 3) to (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply the MobileNetV2 mathematical squish [-1 to 1]
    img_array = preprocess_input(img_array)

    # 5. THE MOMENT OF TRUTH
    print("\nConsulting HemoVision...")
    prediction_matrix = model.predict(img_array, verbose=0)
    
    # Extract the single floating-point number from the prediction matrix
    predicted_hgb = prediction_matrix[0][0]
    
    # 6. The Verdict
    margin_of_error = abs(target_patient['Hgb'] - predicted_hgb)
    
    print("\n======================================")
    print(f" AI PREDICTION: {predicted_hgb:.2f} g/dL")
    print(f" ACTUAL LAB:    {target_patient['Hgb']:.2f} g/dL")
    print("======================================")
    print(f" Margin of Error: {margin_of_error:.2f} points")

if __name__ == "__main__":
    test_hemovision()