import os
import numpy as np
import cv2
from keras.models import load_model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import image_dataset_from_directory
from sklearn.model_selection import train_test_split

# Load dataset (Assuming folder structure: dataset/class_name/image.jpg)
def load_dataset(dataset_path):
    data = []
    labels = []
    class_names = os.listdir(dataset_path)
    class_indices = {name: i for i, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                data.append(img)
                labels.append(class_indices[class_name])
    
    data = np.array(data)
    labels = to_categorical(labels)
    return data, labels, class_names

# Dataset path
data_path = "dataset"

# Load and normalize data
X, y, class_names = load_dataset(data_path)
X = X / 255.0

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save model
model.save("pollen_model.h5")
print("✅ Model training complete and saved as 'pollen_model.h5'")
