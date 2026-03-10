import os
import cv2
import numpy as np
import pickle
from feature_extraction import extract_features
from sklearn.ensemble import RandomForestClassifier

X = []
y = []

# Loop through both categories
for category in ["healthy", "stressed"]:

    path = os.path.join("dataset", category)

    label = 0 if category == "healthy" else 1

    for img_name in os.listdir(path):

        img_path = os.path.join(path, img_name)

        img = cv2.imread(img_path)

        if img is not None:
            features = extract_features(img)
            X.append(features)
            y.append(label)

# Create Random Forest model
model = RandomForestClassifier(n_estimators=100)

# Train model
model.fit(X, y)

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))

print("Model Trained Successfully!")