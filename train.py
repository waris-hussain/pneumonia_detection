import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import matplotlib.pyplot as plt
import kagglehub
import shutil

# --- 1. Configuration ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
MODEL_SAVE_PATH = 'model.h5'

def get_dataset_path():
    """Downloads dataset via kagglehub and returns the path to the training directory."""
    print("Downloading dataset from Kaggle...")
    try:
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print("Dataset downloaded to:", path)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

    # The dataset structure is typically: path/chest_xray/train
    # We need to find where the 'train' folder is.
    
    # Check for direct structure or nested 'chest_xray' folder
    possible_roots = [path, os.path.join(path, "chest_xray")]
    
    train_dir = None
    val_dir = None
    
    for root in possible_roots:
        if os.path.exists(os.path.join(root, "train")):
            train_dir = os.path.join(root, "train")
            val_dir = os.path.join(root, "val")
            if not os.path.exists(val_dir):
                 # Fallback to test if val doesn't exist
                 val_dir = os.path.join(root, "test")
            break
            
    if train_dir:
        print(f"Found training data at: {train_dir}")
        return train_dir, val_dir
    else:
        print("Could not find 'train' directory in the downloaded dataset.")
        return None, None

def build_model():
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,), 
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = False 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x) 
    predictions = Dense(2, activation='softmax')(x) 

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train():
    # --- 2. Data Preparation ---
    train_dir, val_dir = get_dataset_path()
    
    if not train_dir:
        return

    train_datagen = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=20, 
        zoom_range=0.2, 
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    print("Loading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    if val_dir and os.path.exists(val_dir):
        print(f"Loading Validation Data from {val_dir}...")
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
    else:
        print("Warning: No validation data found. Using training data for validation (not ideal but works for assignment).")
        val_generator = train_generator

    # --- 3. Model Training ---
    model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=0.0001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    print("Starting Model Training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )

    # --- 4. Save Model ---
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH, include_optimizer=False)
    
    file_size = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
    print(f"Model saved. Size: {file_size:.2f} MB")

    # --- 5. Plot Results ---
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig('training_results.png')
    print("Training chart saved as 'training_results.png'")

if __name__ == '__main__':
    train()
