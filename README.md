# 🖋️ Handwriting Recognition Web App

A simple Flask-based web application for recognizing handwritten digits (0-9) and letters (A-Z, a-z) drawn on a canvas.  
Built with a CNN trained on the English Handwritten Characters Dataset.

---

## 📋 Features

- Predict digits and letters from freehand canvas drawings.
- Mode selection: Digits or Letters.
- Real-time prediction with confidence scores.
- 280x280 drawing canvas, line width control, clear & recognize buttons.
- Lightweight Flask server for backend processing.
- Responsive, beginner-friendly UI.

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript (AJAX for requests)
- **Backend:** Python Flask
- **Model:** Convolutional Neural Network (Keras)
- **Dataset:** [English Handwritten Characters Dataset](https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset)

---

## 📦 Dataset and Model

- 3,410 images, ~62 classes (0-9, A-Z, a-z).
- Images resized to 28x28 pixels, grayscale.
- Class imbalance handled with class weights.
- Model trained for 10 epochs with augmentation (rotation, zoom).
- Model saved as `handwritten_model.h5`.

---
![Screenshot 2025-04-24 194341](https://github.com/user-attachments/assets/d9484402-6c59-4666-b6ab-6c71e86b010e)
![Screenshot 2025-04-24 194407](https://github.com/user-attachments/assets/575af903-991b-4ae8-b49b-33dd083fe0ec)
---
## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/handwriting-recognition-app.git
   cd handwriting-recognition-app
