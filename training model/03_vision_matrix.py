import cv2
import pandas as pd
import os
import numpy as np

def find_palpebral_image(folder_path):
    """Searches the folder and returns the exact filename containing 'palpebral'."""
    try:
        files = os.listdir(folder_path)
        for file in files:
            if 'palpebral' in file.lower():
                return os.path.join(folder_path, file)
    except FileNotFoundError:
        return None
    return None

def test_vision_pipeline(csv_path="train_set.csv", base_image_dir="."):
    print(f"Loading pipeline checkpoint: {csv_path}...")
    
    # 1. Load the training data
    train_df = pd.read_csv(csv_path)
    
    # 2. Grab the very first patient to test
    test_patient = train_df.iloc[0]
    dataset = test_patient['Dataset']
    number = test_patient['Number']
    hgb_target = test_patient['Hgb']
    
    print(f"\nTarget Patient: {dataset} #{number} (Hgb: {hgb_target})")
    
    # 3. Navigate the hard drive
    folder_path = os.path.join(base_image_dir, dataset, str(number))
    image_path = find_palpebral_image(folder_path)
    
    if not image_path:
        print("Error: Could not locate the image on the hard drive.")
        return

    print(f"Found image at: {image_path}")
    
    # 4. THE EXTRACTION (OpenCV)
    # Read the image from the hard drive into memory
    img_matrix = cv2.imread(image_path)
    
    if img_matrix is None:
        print("Error: OpenCV could not read the image file. It might be corrupted.")
        return

    # 5. THE TRANSLATION
    # OpenCV loads images in BGR (Blue, Green, Red) format by default.
    # TensorFlow and MobileNetV2 expect RGB (Red, Green, Blue). We MUST flip it.
    img_rgb = cv2.cvtColor(img_matrix, cv2.COLOR_BGR2RGB)
    
    # Resize to the exact dimensions MobileNetV2 requires (224x224 pixels)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # 6. The Proof
    print("\n--- Pipeline Success ---")
    print(f"Final Matrix Shape: {img_resized.shape}")
    print("This means: (Height=224, Width=224, Color Channels=3)")
    print("\nHere is what the top-left pixel looks like mathematically (R, G, B):")
    print(img_resized[0, 0])

if __name__ == "__main__":
    test_vision_pipeline()