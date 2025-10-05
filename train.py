import ArcFace
import argparse
import cv2
import glob
import numpy as np
import keras
from keras import layers, Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow as tf
import pickle
from keras.layers import Dropout, BatchNormalization
from keras.optimizers import Adam

# ============================================================
# Argument Parser — Command-line configuration
# ============================================================
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, default='Norm',
                help="Path to normalized dataset directory")
ap.add_argument("-o", "--save", type=str, default='models/model.keras',
                help="Path to save trained model (.keras format)")
ap.add_argument("-l", "--le", type=str, default='models/le.pickle',
                help="Path to save label encoder pickle file")
ap.add_argument("-b", "--batch_size", type=int, default=16,
                help="Batch size for training")
ap.add_argument("-e", "--epochs", type=int, default=800,
                help="Number of epochs for training")

args = vars(ap.parse_args())
path_to_dir = args["dataset"]
checkpoint_path = args['save']

# ============================================================
# Load Pre-trained ArcFace Model
# ============================================================
model = ArcFace.loadModel()
model.load_weights("arcface_weights.h5")

print("ArcFace expects input shape:", model.layers[0].input_shape[0][1:])
print("Face embeddings dimensionality:",
      model.layers[-1].output_shape[1:], "dimensions")

target_size = model.layers[0].input_shape[0][1:3]
print('Target input size for face images:', target_size)

# ============================================================
# Generate Face Embeddings for Each Image
# ============================================================
x = []  # List to store face embeddings (feature vectors)
y = []  # List to store corresponding labels (person names)

names = sorted(os.listdir(path_to_dir))
class_number = len(names)

for name in names:
    img_list = sorted(glob.glob(os.path.join(path_to_dir, name) + '/*'))

    for img_path in img_list:
        img = cv2.imread(img_path)
        img_resize = cv2.resize(img, target_size)

        # Convert image to array and normalize pixel values
        img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_norm = img_pixels / 255.0

        # Pass image through ArcFace to get 512-d embedding
        img_embedding = model.predict(img_norm)[0]

        x.append(img_embedding)
        y.append(name)
        print(f'[INFO] Embedded {img_path}')

    print(f'[INFO] Completed embeddings for class: {name}')

print('[INFO] Image embedding generation completed.')

# ============================================================
# Prepare DataFrame and Encode Labels
# ============================================================
df = pd.DataFrame(x, columns=np.arange(512))
x = df.copy().astype('float64')

# Encode class labels
le = LabelEncoder()
labels = le.fit_transform(y)
labels = tf.keras.utils.to_categorical(labels, class_number)

# Split data into training and validation sets
x_train, x_test, y_train, y_test = train_test_split(
    x, labels, test_size=0.2, random_state=0
)

# ============================================================
# Define Classification Neural Network
# ============================================================
model = Sequential([
    layers.Dense(1024, activation='relu', input_shape=[512]),
    BatchNormalization(),
    Dropout(0.4),

    layers.Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    layers.Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    layers.Dense(class_number, activation='softmax')
])

print('Model Summary:')
model.summary()

# ============================================================
# Compile Model
# ============================================================
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# Define Callbacks: Checkpoint, EarlyStopping, LR Scheduler
# ============================================================
checkpoint = keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

earlystopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=200
)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    verbose=1
)

# ============================================================
# Train Model
# ============================================================
print('[INFO] Model training started ...')

history = model.fit(
    x_train, y_train,
    epochs=args['epochs'],
    batch_size=args['batch_size'],
    validation_data=(x_test, y_test),
    callbacks=[checkpoint, earlystopping, lr_scheduler]
)

print('[INFO] Model training completed.')
print(f'[INFO] Best model saved at: {checkpoint_path}')

# ============================================================
# Save Label Encoder for Later Inference
# ============================================================
with open(args["le"], "wb") as f:
    f.write(pickle.dumps(le))

print('[INFO] Label encoder saved as models/le.pickle')

# ============================================================
# Plot Training Curves (Loss & Accuracy)
# ============================================================
metric_loss = history.history['loss']
metric_val_loss = history.history['val_loss']
metric_accuracy = history.history['accuracy']
metric_val_accuracy = history.history['val_accuracy']
epochs_range = range(len(metric_loss))

# ---------- Plot Loss ----------
plt.figure()
plt.plot(epochs_range, metric_loss, 'blue', label='Training Loss')
plt.plot(epochs_range, metric_val_loss, 'red', label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

if os.path.exists('loss.png'):
    os.remove('loss.png')
plt.savefig('loss.png', bbox_inches='tight')
print('[INFO] Loss plot saved as loss.png')

# ---------- Plot Accuracy ----------
plt.figure()
plt.plot(epochs_range, metric_accuracy, 'green', label='Training Accuracy')
plt.plot(epochs_range, metric_val_accuracy, 'orange', label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

if os.path.exists('accuracy.png'):
    os.remove('accuracy.png')
plt.savefig('accuracy.png', bbox_inches='tight')
print('[INFO] Accuracy plot saved as accuracy.png')
