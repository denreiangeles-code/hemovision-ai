import tensorflow as tf
from keras.applications import MobileNetV2
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam

# Import the pipeline you built earlier
from data_loader import build_data_pipelines

def build_hemovision_model():
    print("Downloading MobileNetV2 Base Architecture...")
    
    # 1. The Pre-trained Brain (Transfer Learning)
    # We drop the top layer because MobileNet was built to classify 1000 objects. 
    # We only care about predicting one number: Hemoglobin.
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False, 
        weights='imagenet' # Start with Google's pre-trained edge-detection weights
    )
    
    # Freeze the base so we don't accidentally destroy its foundational knowledge
    base_model.trainable = False 

    # 2. The Custom Regression Head
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(), # Flattens the 3D matrices into a 1D array
        Dropout(0.2),             # Drops 20% of connections randomly to prevent overfitting
        
        # The Output Node: A single linear node to predict a continuous float (e.g., 12.5)
        Dense(1, activation='linear') 
    ])

    # 3. The Compiler
    # MAE (Mean Absolute Error) is perfect here. 
    # If the MAE is 1.0, it means the AI's guess is off by exactly 1.0 g/dL of Hemoglobin.
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_absolute_error',
        metrics=['mean_squared_error']
    )
    
    return model

if __name__ == "__main__":
    # 1. Mount the data
    train_gen, val_gen = build_data_pipelines()
    
    # 2. Build the model
    model = build_hemovision_model()
    model.summary() # Prints the architecture blueprint
    
    # 3. THE TRAINING LOOP
    print("\n--- Initiating HemoVision Training ---")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10, # The Dry Run limit
        verbose="auto"  # Shows the progress bar for every batch
    )
    
    print("\nTraining Complete! Saving weights...")
    model.save('hemovision_poc_v1.keras')