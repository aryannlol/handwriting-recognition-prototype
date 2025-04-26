import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

# Parameters
IMG_SIZE = (28, 28)  # Standard for handwritten datasets
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = 'dataset/Img'  # Path to images (adjust if no Img/ subfolder)
CSV_FILE = 'dataset/english.csv'  # Path to CSV
MODEL_PATH = 'handwritten_model.h5'

# Load data from CSV and images
def load_data():
    print("Loading english.csv...")
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"Error reading {CSV_FILE}: {e}")
        return None, None, None, None

    if 'image' not in df.columns or 'label' not in df.columns:
        print("english.csv must have 'image' and 'label' columns")
        return None, None, None, None

    images = []
    labels = []
    class_names = sorted(df['label'].astype(str).unique())
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)

    print(f"Found {num_classes} classes: {class_names}")
    for _, row in df.iterrows():
        filename = row['image']
        label = str(row['label'])
        
        # Handle possible Img/ prefix in CSV
        if filename.startswith('Img/'):
            filename = filename.replace('Img/', '')
        
        img_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(img_path):
            print(f"Skipping {filename}: File not found")
            continue

        try:
            img = load_img(img_path, color_mode='grayscale', target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(class_indices[label])
            print(f"Loaded {img_path}: Label {label}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if not images:
        print("No valid images loaded")
        return None, None, None, None

    images = np.array(images)
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    print(f"Loaded {len(images)} images")
    return images, labels, class_names, class_indices

# Train the model
def train_model(images, labels, num_classes):
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

    return model, history

# GUI for drawing and prediction
class DrawingApp:
    def __init__(self, root, model, class_names, class_indices):
        self.root = root
        self.model = model
        self.class_names = class_names
        self.class_indices = class_indices
        self.root.title("Draw Digit or Letter")

        self.canvas = tk.Canvas(root, width=200, height=200, bg='white')
        self.canvas.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.start_draw)

        tk.Button(root, text="Predict", command=self.predict).pack(pady=5)
        tk.Button(root, text="Clear", command=self.clear).pack(pady=5)

        self.result_label = tk.Label(root, text="Prediction: None")
        self.result_label.pack(pady=5)

        self.canvas_plot = tk.Canvas(root, width=300, height=150)
        self.canvas_plot.pack(pady=10)

        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

    def draw(self, event):
        x, y = event.x, event.y
        if self.last_x is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill='black', width=5)
            self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=5)
        self.last_x, self.last_y = x, y

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Prediction: None")
        self.canvas_plot.delete("all")

    def predict(self):
        img = self.image.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        predictions = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        class_label = self.class_names[predicted_class]
        confidence = predictions[0][predicted_class]

        self.result_label.config(text=f"Prediction: {class_label} ({confidence:.2%})")

        plt.figure(figsize=(6, 3))
        plt.bar(self.class_names, predictions[0], alpha=0.5)
        plt.xticks(rotation=45)
        plt.ylabel('Probability')
        plt.title('Prediction Probabilities')
        plt.tight_layout()
        plt.savefig('prediction_probs.png')
        plt.close()

        img = Image.open('prediction_probs.png')
        img = img.resize((300, 150))
        self.photo = tk.PhotoImage(file='prediction_probs.png')
        self.canvas_plot.create_image(0, 0, anchor='nw', image=self.photo)

# Main execution
if __name__ == "__main__":
    images, labels, class_names, class_indices = load_data()
    if images is None:
        print("Failed to load data. Check dataset/Img/ and english.csv.")
        exit()

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model, _ = train_model(images, labels, len(class_names))

    root = tk.Tk()
    app = DrawingApp(root, model, class_names, class_indices)
    root.mainloop()