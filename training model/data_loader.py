import pandas as pd
import os
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

def get_image_path(row, base_dir="."):
    """Rebuilds the path and physically verifies the image isn't corrupted."""
    folder_path = os.path.join(base_dir, row['Dataset'], str(row['Number']))
    try:
        for file in os.listdir(folder_path):
            # 1. Ignore invisible Mac files right away
            if file.startswith('._'):
                continue
                
            # 2. THE NEW FILTER: Exact string match for pure palpebral masks
            file_lower = file.lower()
            if 'palpebral' in file_lower and 'forniceal' not in file_lower:
                
                full_path = os.path.join(folder_path, file)
                
                # 3. THE BOUNCER: Verify the physical file isn't corrupted
                try:
                    with Image.open(full_path) as img:
                        img.verify() 
                    return full_path # If it survives the bouncer, return it!
                except Exception:
                    # If Pillow crashes, the file is corrupted. Skip it.
                    continue
    except FileNotFoundError:
        return None
    return None

def build_data_pipelines(train_csv='train_set.csv', val_csv='val_set.csv', batch_size=16):
    print("Initializing TensorFlow Data Generators...")
    
    # 1. Load the CSV Checkpoints
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    # 2. Map the exact file paths
    train_df['filepath'] = train_df.apply(get_image_path, axis=1)
    val_df['filepath'] = val_df.apply(get_image_path, axis=1)

    # Drop any rows where the image was corrupted or missing
    train_df = train_df.dropna(subset=['filepath'])
    val_df = val_df.dropna(subset=['filepath'])
    
    # 3. THE TRAINING ENGINE (Augmentation + Normalization)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, # The MobileNetV2 Math Fix [-1 to 1]
        rotation_range=15,       # Tilt the image up to 15 degrees
        horizontal_flip=True,    # Flip left/right
        zoom_range=0.1,          # Zoom in/out by 10%
        fill_mode='constant',
        cval=0                   # If rotation creates empty space, fill it with black (0)
    )
    
    # 4. THE VALIDATION ENGINE (Normalization ONLY)
    # CRITICAL: We NEVER augment the validation set. It must remain a pure test.
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    # 5. Build the Iterators (The Batchers)
    print("Mounting Training Data:")
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='Hgb',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='raw',       # 'raw' means our target (Hgb) is a continuous number
        batch_size=batch_size,
        shuffle=True            # Shuffle the homework so the model doesn't memorize the order
    )
    
    print("\nMounting Validation Data:")
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filepath',
        y_col='Hgb',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='raw',
        batch_size=batch_size,
        shuffle=False           # Never shuffle the practice exam
    )
    
    return train_generator, val_generator

if __name__ == "__main__":
    train_gen, val_gen = build_data_pipelines()
    
    # Prove it works by grabbing exactly one batch of 16 images
    images, hemoglobin_targets = next(train_gen)
    
    print("\n--- Pipeline Hand-off Successful ---")
    print(f"Batch Shape: {images.shape} -> (Batch_Size, Height, Width, Channels)")
    print(f"Target Shape: {hemoglobin_targets.shape} -> (Batch_Size,)")