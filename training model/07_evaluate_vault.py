import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

def evaluate_vault():
    print("Booting up HemoVision AI...")
    model = tf.keras.models.load_model('hemovision_fold_1_best.keras')
    test_df = pd.read_csv('test_set.csv')
    
    print(f"Opening the Vault... found {len(test_df)} patients.")
    errors = []
    successful_tests = 0

    for index, patient in test_df.iterrows():
        folder_path = os.path.join(".", patient['Dataset'], str(patient['Number']))
        img_path = None
        
        # The Bouncer (Ignore hidden Mac files)
        try:
            for file in os.listdir(folder_path):
                if 'palpebral' in file.lower() and not file.startswith('._'):
                    img_path = os.path.join(folder_path, file)
                    break
        except FileNotFoundError:
            continue

        if not img_path:
            continue

        # Translation
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))
            img_array = preprocess_input(np.expand_dims(np.array(img), axis=0))
        except Exception:
            continue # Skip corrupted images

        # Prediction (Read-Only Mode)
        predicted_hgb = model.predict(img_array, verbose=0)[0][0]
        actual_hgb = patient['Hgb']
        
        # Calculate how far off the AI was
        margin_of_error = abs(actual_hgb - predicted_hgb)
        errors.append(margin_of_error)
        successful_tests += 1

    # The Final Grade
    if successful_tests > 0:
        average_mae = sum(errors) / len(errors)
        print("\n======================================")
        print(" 🏆 FINAL VAULT EVALUATION 🏆")
        print("======================================")
        print(f" Images Tested:   {successful_tests}")
        print(f" Overall Average Error: {average_mae:.2f} g/dL")
        print("======================================")

if __name__ == "__main__":
    evaluate_vault()