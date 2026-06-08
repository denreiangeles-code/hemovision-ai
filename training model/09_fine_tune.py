import tensorflow as tf
from keras.optimizers import Adam
from data_loader import build_data_pipelines

def fine_tune_best_model():
    print("Loading the Champion Model (Fold 3)...")
    
    # 1. Load your best performing K-Fold model
    try:
        model = tf.keras.models.load_model('hemovision_fold_3_best.keras')
    except Exception:
        print("Error: Could not find the Fold 3 model. Check the exact filename!")
        return

    # 2. Extract the locked MobileNetV2 Base
    # We grab it by the exact name Keras assigned it in your summary earlier
    base_model = model.get_layer('mobilenetv2_1.00_224')
    
    # 3. UNLOCK THE BASE
    base_model.trainable = True
    
    # 4. Freeze everything EXCEPT the top 20 layers
    # We want to keep the foundational edge-detection, but rewire the high-level pattern recognition
    print(f"Base model has {len(base_model.layers)} layers. Unfreezing the top 20...")
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # 5. RECOMPILE WITH A MICROSCOPIC LEARNING RATE
    # CRITICAL: If you use a normal learning rate here, the AI will "forget" everything 
    # and destroy the pre-trained weights. We use 1e-5 (0.00001) for tiny, surgical updates.
    model.compile(
        optimizer=Adam(learning_rate=1e-5), 
        loss='mean_absolute_error',
        metrics=['mean_squared_error']
    )
    
    model.summary()

    # 6. Mount the standard data pipeline
    train_gen, val_gen = build_data_pipelines()

    # 7. The Surgical Training Loop
    print("\n--- Initiating Surgical Fine-Tuning ---")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15, 
        verbose="auto"
    )
    
    print("\nFine-Tuning Complete! Saving the ultimate model...")
    model.save('hemovision_ultimate.keras')

if __name__ == "__main__":
    fine_tune_best_model()