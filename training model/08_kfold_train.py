import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint

from data_loader import get_image_path
from train_model import build_hemovision_model 


def run_kfold_training(k=5):
    print(f"--- Initiating {k}-Fold Cross-Validation ---")
    
    # 1. Recombine the Development Pool
    # We bring the train and val sets back together so we can slice them dynamically.
    # The Vault (test_set.csv) remains safely locked away.
    df_train = pd.read_csv('train_set.csv')
    df_val = pd.read_csv('val_set.csv')
    master_df = pd.concat([df_train, df_val], ignore_index=True)
    
    # Run the Pillow Bouncer to drop corrupted files upfront
    master_df['filepath'] = master_df.apply(get_image_path, axis=1)
    master_df = master_df.dropna(subset=['filepath']).reset_index(drop=True)
    
    print(f"Total Valid Images for K-Fold: {len(master_df)}")

    # 2. The K-Fold Slicer
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Keep track of the scores
    fold_scores = []

    # 3. THE K-FOLD LOOP
    for fold, (train_idx, val_idx) in enumerate(kf.split(master_df), 1):
        print(f"\n======================================")
        print(f" 🚀 STARTING FOLD {fold}/{k}")
        print(f"======================================")
        
        # Slice the dataframe into this fold's specific Train/Val chunks
        fold_train_df = master_df.iloc[train_idx]
        fold_val_df = master_df.iloc[val_idx]
        
        # 4. The Generators (Augment Training, Pure Validation)
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=15,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='constant', cval=0
        )
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        
        train_gen = train_datagen.flow_from_dataframe(
            dataframe=fold_train_df, x_col='filepath', y_col='Hgb',
            target_size=(224, 224), class_mode='raw', batch_size=16, shuffle=True
        )
        val_gen = val_datagen.flow_from_dataframe(
            dataframe=fold_val_df, x_col='filepath', y_col='Hgb',
            target_size=(224, 224), class_mode='raw', batch_size=16, shuffle=False
        )

        # 5. Initialize a fresh brain for this fold
        model = build_hemovision_model()
        
        # 6. The Shields (Early Stopping & Checkpointing)
        # Stop training if val_loss doesn't improve for 5 epochs
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        # Save the absolute best version of this fold's model
        checkpoint = ModelCheckpoint(f'hemovision_fold_{fold}_best.keras', monitor='val_loss', save_best_only=True)

        # 7. Train the Fold
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=20, # We can safely raise this because Early Stopping will hit the brakes if needed
            verbose="auto",
            callbacks=[early_stop, checkpoint]
        )
        
        # Record the best MAE for this fold
        best_mae = min(history.history['val_loss'])
        fold_scores.append(best_mae)
        print(f"✅ Fold {fold} Complete. Best MAE: {best_mae:.2f} g/dL")

    # 8. The Final Verdict
    average_kfold_mae = np.mean(fold_scores)
    print("\n======================================")
    print(" 🏆 K-FOLD CROSS-VALIDATION COMPLETE 🏆")
    print("======================================")
    for i, score in enumerate(fold_scores, 1):
        print(f" Fold {i}: {score:.2f} g/dL")
    print(f" -> OVERALL AVERAGE MAE: {average_kfold_mae:.2f} g/dL")

if __name__ == "__main__":
    run_kfold_training()