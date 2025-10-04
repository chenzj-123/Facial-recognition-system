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
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam

# Argument parser setup
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, default='Norm',
                help="path to Norm/dir")
ap.add_argument("-o", "--save", type=str, default='models/model.keras',
                help="path to save .keras model, e.g. dir/model.keras")
ap.add_argument("-l", "--le", type=str, default='models/le.pickle',
                help="path to label encoder")
ap.add_argument("-b", "--batch_size", type=int, default=16,
                help="batch size for model training")
ap.add_argument("-e", "--epochs", type=int, default=800,
                help="epochs for model training")

args = vars(ap.parse_args())
path_to_dir = args["dataset"]
checkpoint_path = args['save']

# Load ArcFace model
model = ArcFace.loadModel()
model.load_weights("arcface_weights.h5")
print("ArcFace expects ", model.layers[0].input_shape[0][1:], " inputs")
print("and it represents faces as ",
      model.layers[-1].output_shape[1:], " dimensional vectors")
target_size = model.layers[0].input_shape[0][1:3]
print('target_size: ', target_size)

# Extract face embeddings
x = []
y = []

names = os.listdir(path_to_dir)
names = sorted(names)
class_number = len(names)

for name in names:
    img_list = glob.glob(os.path.join(path_to_dir, name) + '/*')
    img_list = sorted(img_list)

    for img_path in img_list:
        img = cv2.imread(img_path)
        img_resize = cv2.resize(img, target_size)
        img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_norm = img_pixels / 255.0  # normalize to [0, 1]
        img_embedding = model.predict(img_norm)[0]

        x.append(img_embedding)
        y.append(name)
        print(f'[INFO] Embedding {img_path}')
    print(f'[INFO] Completed {name} Part')
print('[INFO] Image Data Embedding Completed...')

# Convert to DataFrame
df = pd.DataFrame(x, columns=np.arange(512))
x = df.copy().astype('float64')

# Label encoding
le = LabelEncoder()
labels = le.fit_transform(y)
labels = tf.keras.utils.to_categorical(labels, class_number)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, labels, test_size=0.2, random_state=0)

# Neural network model
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

# Model summary
print('Model Summary: ', model.summary())

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model checkpoint with native Keras format
checkpoint = keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)
earlystopping = keras.callbacks.EarlyStopping(
  monitor='val_accuracy', patience=100)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    verbose=1
)

print('[INFO] Model Training Started ...')
history = model.fit(x_train, y_train,
                    epochs=args['epochs'],
                    batch_size=args['batch_size'],
                    validation_data=(x_test, y_test),
                    callbacks=[checkpoint,earlystopping, lr_scheduler])
print('[INFO] Model Training Completed')
print(f'[INFO] Model Successfully Saved in /{checkpoint_path}')

# Save label encoder
with open(args["le"], "wb") as f:
    f.write(pickle.dumps(le))
print('[INFO] Successfully Saved models/le.pickle')

# Plot training metrics separately

# Get training data
metric_loss = history.history['loss']
metric_val_loss = history.history['val_loss']
metric_accuracy = history.history['accuracy']
metric_val_accuracy = history.history['val_accuracy']
epochs_range = range(len(metric_loss))

# Plot Loss
plt.figure()
plt.plot(epochs_range, metric_loss, 'blue', label='loss')
plt.plot(epochs_range, metric_val_loss, 'red', label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

if os.path.exists('loss.png'):
    os.remove('loss.png')
plt.savefig('loss.png', bbox_inches='tight')
print('[INFO] Successfully Saved loss.png')

# Plot Accuracy
plt.figure()
plt.plot(epochs_range, metric_accuracy, 'green', label='accuracy')
plt.plot(epochs_range, metric_val_accuracy, 'orange', label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

if os.path.exists('accuracy.png'):
    os.remove('accuracy.png')
plt.savefig('accuracy.png', bbox_inches='tight')
print('[INFO] Successfully Saved accuracy.png')

