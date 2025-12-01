import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

print("[INFO] Loading CIFAR-10 dataset...")

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"[INFO] Training data shape: {x_train.shape}")
print(f"[INFO] Test data shape: {x_test.shape}")
print(f"[INFO] Training labels shape: {y_train.shape}")

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train_categorical = keras.utils.to_categorical(y_train, 10)
y_test_categorical = keras.utils.to_categorical(y_test, 10)

print(f"[INFO] Example original label: {y_train[0][0]}")
print(f"[INFO] Example one-hot label: {y_train_categorical[0]}")


print("[INFO] Building the CNN model...")

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                       input_shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

print("[INFO] Compiling the model...")
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("[INFO] Starting model training...")
history = model.fit(
    x_train, y_train_categorical,
    epochs=15,
    batch_size=64,
    validation_data=(x_test, y_test_categorical)
)

print("[INFO] Training finished.")

print("[INFO] Evaluating model on test data...")
test_loss, test_acc = model.evaluate(x_test, y_test_categorical, verbose=2)
print(f"\n[RESULT] Test Accuracy: {test_acc * 100:.2f}%")
print(f"[RESULT] Test Loss: {test_loss:.4f}")

print("\n[INFO] Making a prediction on a sample test image...")

image_index = np.random.randint(0, len(x_test))
sample_image = x_test[image_index]
actual_label_index = y_test[image_index][0]
actual_label_name = class_names[actual_label_index]

plt.imshow(sample_image)
plt.title(f"Actual Label: {actual_label_name}")
plt.axis('off')
plt.show(block=False)

image_for_prediction = np.expand_dims(sample_image, axis=0)

predictions_vector = model.predict(image_for_prediction)
predicted_label_index = np.argmax(predictions_vector)
predicted_label_name = class_names[predicted_label_index]

print(f"\n--- Prediction Result (Image {image_index}) ---")
print(f"Actual Class: \t{actual_label_name} (Index {actual_label_index})")
print(f"Predicted Class: \t{predicted_label_name} (Index {predicted_label_index})")
print(f"Confidence: \t{np.max(predictions_vector) * 100:.2f}%")

print("\nFull Prediction Probabilities:")
for i, class_name in enumerate(class_names):
    print(f"  {class_name.ljust(12)}: {predictions_vector[0][i] * 100:.2f}%")

plt.show()