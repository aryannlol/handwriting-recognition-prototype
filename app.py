import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
from sklearn.model_selection import train_test_split
import pickle
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parameters
IMG_SIZE = (28, 28)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = os.path.join('dataset', 'Img')  # Cross-platform path
CSV_FILE = os.path.join('dataset', 'english.csv')
MODEL_PATH = 'handwritten_model.h5'
METADATA_PATH = 'metadata.pkl'

# Load data from CSV and images (only for training)
def load_data():
    logging.info(f"Loading {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        logging.error(f"Error reading {CSV_FILE}: {e}")
        return None, None, None, None

    if 'image' not in df.columns or 'label' not in df.columns:
        logging.error("english.csv must have 'image' and 'label' columns")
        return None, None, None, None

    images = []
    labels = []
    class_names = sorted(df['label'].astype(str).str.upper().unique())  # Normalize to uppercase
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)

    logging.info(f"Found {num_classes} classes: {class_names}")
    for _, row in df.iterrows():
        filename = row['image']
        label = str(row['label']).upper()  # Normalize to uppercase
        
        if filename.startswith('Img/') or filename.startswith('Img\\'):
            filename = filename.replace('Img/', '').replace('Img\\', '')
        
        img_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(img_path):
            logging.warning(f"Skipping {filename}: File not found")
            continue

        try:
            img = load_img(img_path, color_mode='grayscale', target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(class_indices[label])
            logging.info(f"Loaded {img_path}: Label {label}")
        except Exception as e:
            logging.error(f"Error loading {filename}: {e}")

    if not images:
        logging.error("No valid images loaded")
        return None, None, None, None

    images = np.array(images)
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    logging.info(f"Loaded {len(images)} images")
    return images, labels, class_names, class_indices

# Train the model
def train_model(images, labels, num_classes):
    logging.info("Training model...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        verbose=1
    )

    model.save(MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_metrics.png')
    plt.close()
    logging.info("Training metrics plot saved to training_metrics.png")

    return model, history

# Load model and metadata
def load_model_and_metadata():
    global model, class_names, class_indices
    if os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH):
        logging.info(f"Found {MODEL_PATH} and {METADATA_PATH}. Loading...")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(METADATA_PATH, 'rb') as f:
                metadata = pickle.load(f)
            class_names = metadata['class_names']
            class_indices = metadata['class_indices']
            logging.info(f"Loaded model and metadata: {len(class_names)} classes")
        except Exception as e:
            logging.error(f"Error loading model/metadata: {e}. Will retrain.")
            return False
    else:
        logging.info(f"Missing {MODEL_PATH} or {METADATA_PATH}. Will train new model.")
        return False
    return True

# Initialize model and metadata
logging.info("Starting app...")
if not load_model_and_metadata():
    logging.info("Training required...")
    images, labels, class_names, class_indices = load_data()
    if images is None:
        logging.error("Failed to load data. Check dataset/Img/ and english.csv.")
        exit(1)
    model, _ = train_model(images, labels, len(class_names))
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump({'class_names': class_names, 'class_indices': class_indices}, f)
    logging.info(f"Saved metadata to {METADATA_PATH}")
else:
    logging.info("App ready with loaded model and metadata.")

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json()
    image_data = data['image']
    mode = data['mode']

    try:
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize(IMG_SIZE)
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        predictions = model.predict(img_array, verbose=0)[0]
        
        if mode == 'digit':
            valid_indices = [class_indices[str(i)] for i in range(10) if str(i) in class_indices]
            valid_names = [str(i) for i in range(10) if str(i) in class_indices]
        else:  # letter
            valid_indices = [class_indices[chr(i)] for i in range(65, 91) if chr(i) in class_indices]
            valid_names = [chr(i) for i in range(65, 91) if chr(i) in class_indices]

        valid_probs = predictions[valid_indices]
        predicted_idx = np.argmax(valid_probs)
        predicted_class = valid_names[predicted_idx]
        confidence = float(valid_probs[predicted_idx]) * 100

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({
            'prediction': 'Error',
            'confidence': 0
        }), 500

if __name__ == '__main__':
    app.run(debug=True)