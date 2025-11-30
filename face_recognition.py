import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

DATA_DIR = 'dataset/train'  
IMG_SIZE = (224, 224)       
BATCH_SIZE = 32


train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)


class_names = train_ds.class_names
print(f"{class_names}") 




model = models.Sequential([

    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),

    # Convolutional Block 1
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Convolutional Block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Convolutional Block 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten & Dense Layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Prevents overfitting
    
    # Output Layer 
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15  
)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.show()

import numpy as np
from tensorflow.keras.preprocessing import image

# Path to a new unseen image
img_path = 'test_image.jpg' 

# Load and Resize
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) 

# Predict
predictions = model.predict(img_array)
score = predictions[0][0]

print(f"Raw Score: {score}")

if score > 0.5:
    print(f"It's {class_names[1]} ({score*100:.2f}% confidence)")
else:
    print(f"It's {class_names[0]} ({(1-score)*100:.2f}% confidence)")