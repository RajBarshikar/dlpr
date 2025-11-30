import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('fashion-mnist_train.csv')

X = df.drop('label', axis=1)
y = df['label']

X = X / 255.0

X = X.values
y = y.values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

num_classes = len(np.unique(y))
print(f"Number of classes found: {num_classes}")

model = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax') 
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val)
)

test_loss, test_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {test_acc*100:.2f}%")

test_df = pd.read_csv('fashion-mnist_test.csv')

X_test = test_df.drop('label', axis=1).values
y_test = test_df['label'].values

X_test = X_test / 255.0

print("Evaluating on test data...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

print(f"\n--------------------------------")
print(f"Final Test Accuracy: {test_acc*100:.2f}%")
print(f"--------------------------------\n")

predictions = model.predict(X_test)
y_pred_classes = np.argmax(predictions, axis=1)

print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

import matplotlib.pyplot as plt

# 1. Select the first image from the test set
image_index = 5
test_image = X_test[image_index]
true_label = y_test[image_index]

# 2. Reshape for prediction (Model expects shape (1, 784))
input_image = test_image.reshape(1, 784)

# 3. Get Prediction
prediction_probs = model.predict(input_image)
predicted_label = np.argmax(prediction_probs)

# 4. Plot the image
plt.figure(figsize=(4, 4))
plt.imshow(test_image.reshape(28, 28), cmap='gray')
plt.title(f"True Label: {true_label}\nPredicted Label: {predicted_label}")
plt.axis('off') # Remove axis ticks
plt.show()

print(f"Model Confidence: {np.max(prediction_probs) * 100:.2f}%")