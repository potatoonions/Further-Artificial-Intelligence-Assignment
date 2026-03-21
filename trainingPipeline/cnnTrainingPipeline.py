# This file is a safecopy of the Google collab to build and 
# train the CNN used in this project. This is not intended 
# to be ran locally, but into the collab, remotely.

##############################################
#########  STEP 1: Dependency Setup  #########
##############################################

# Install dependencies using this command:
#!pip install -q kagglehub opencv-python tensorflow matplotlib scikit-learn

import kagglehub
import os

# Download the dataset
path = kagglehub.dataset_download("robertmifsud/resized-reduced-gz2-images")

# Show what's inside
for root, dirs, files in os.walk(path):
    print("📁", root)
    for d in dirs:
        print("   └──", d)

dataset_path = "/root/.cache/kagglehub/datasets/robertmifsud/resized-reduced-gz2-images/versions/2/images_E_S_SB_299x299_a_03/images_E_S_SB_299x299_a_03_train"

##############################################
##########  STEP 2: Dataset Setup  ###########
##############################################

from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = 256 #299
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

##############################################
######## STEP 3: Building the CNN  ###########
##############################################

from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(img_size, img_size, 3)),

    layers.Conv2D(32, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2,2),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')  # 3 galaxy types: spiral, elliptical, irregular
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

##############################################
######  STEP 4: Training the Model   #########
##############################################

history = model.fit(
    train_generator,
    steps_per_epoch=200,  # instead of 3011
    epochs=20,
    validation_data=val_generator,
    validation_steps=30
)

##############################################
#### STEP 5: Visualizing Training Results  ###
##############################################

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('AI Learning Progress YAYYYYYYY✨')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

##############################################
###### STEP 6: Predicting one Image   ########
##############################################

import numpy as np
from tensorflow.keras.preprocessing import image
import random

# Choose a random image from validation folder
import glob
img_paths = glob.glob(dataset_path + '/*/*')
rand_img_path = random.choice(img_paths)
print("🔍 Predicting: ", rand_img_path)

img = image.load_img(rand_img_path, target_size=(img_size, img_size))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.0

prediction = model.predict(x)
label_index = np.argmax(prediction)
class_names = list(train_generator.class_indices.keys())

plt.imshow(img)
plt.title(f"👁️ AI thinks this is: {class_names[label_index]}\nConfidence: {prediction[0][label_index]*100:.1f}%")
plt.axis("off")
plt.show()

##############################################
############### STEP 7: Demo   ###############
##############################################

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i in range(5):
    rand_img_path = random.choice(img_paths)
    img = image.load_img(rand_img_path, target_size=(img_size, img_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    prediction = model.predict(x)
    label_index = np.argmax(prediction)
    class_name = class_names[label_index]
    confidence = prediction[0][label_index] * 100

    axes[i].imshow(img)
    axes[i].set_title(f"{class_name}\n{confidence:.1f}%")
    axes[i].axis("off")

plt.suptitle("🌌 Galaxy Classifier AI | Galaxy Shape Predictions")
plt.show()

##############################################
######  STEP 8: Exporting the Model   ########
##############################################

import json
from google.colab import files

EXPORT_DIR = "/content/galaxy_cnn_export"
os.makedirs(EXPORT_DIR, exist_ok=True)

# Save model in Keras native format
model_path = os.path.join(EXPORT_DIR, "galaxy_cnn.keras")
model.save(model_path)
print(f"Model saved to: {model_path}")

# Save class index mapping so the project knows label order
class_indices = train_generator.class_indices          # e.g. {'E': 0, 'S': 1, 'SB': 2}
class_labels  = {v: k for k, v in class_indices.items()}  # flip to {0: 'E', 1: 'S', 2: 'SB'}

metadata = {
    "class_indices": class_indices,
    "class_labels":  class_labels,
    "img_size":      img_size,
    "input_shape":   [img_size, img_size, 3],
    "num_classes":   len(class_indices),
}

metadata_path = os.path.join(EXPORT_DIR, "model_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to: {metadata_path}")

# Download both files to your local machine
files.download(model_path)
files.download(metadata_path)

print("\nExport complete. Place galaxy_cnn.keras and model_metadata.json in your project root.")
