import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
MODEL_PATH = "best_model.h5"

# Load the trained model
model = load_model(MODEL_PATH)

# Define category labels (ensure this matches the dataset categories)
CATEGORY_LABELS = {
    0: "speed limit 20",
    1: "Speed Limit 30",
    2: "speed limit 50",
    3: "Speed limit 60",
    4: "Speed limit 70",
    5: "Speed limit 80",
    9: "No Overtaking",
    14: "Stop Sign",
    17: "No Entry",
    40: "RoundAbout",
}


def predict_image(image_path):
    """
    Takes an image path and returns a prediction (category, sign name, probability).
    """
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(image)
    predicted_category = np.argmax(prediction)
    probability = float(np.max(prediction))

    # Get sign name
    sign_name = CATEGORY_LABELS.get(predicted_category, "Unknown")

    return predicted_category, sign_name, probability


def upload_and_predict():
    """Handles file upload and displays the predicted traffic sign."""
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Get prediction
    category, sign_name, probability = predict_image(file_path)

    # Display result
    result_label.config(text=f"Prediction: {sign_name} ({category})\nConfidence: {probability:.3f}")


# Tkinter GUI
root = tk.Tk()
root.title("Traffic Sign Predictor")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(pady=10)

btn_upload = tk.Button(frame, text="Upload Image", command=upload_and_predict)
btn_upload.pack()

result_label = tk.Label(frame, text="Upload an image to predict", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
