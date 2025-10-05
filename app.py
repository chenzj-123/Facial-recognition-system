import streamlit as st
import cv2
import os
from my_utils import alignment_procedure
from mtcnn import MTCNN
import glob
import ArcFace
import numpy as np
import keras
from keras import layers, Sequential
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras.layers import Dropout, BatchNormalization
from keras.optimizers import Adam

# ============================================================
# Streamlit Interface — Title & Global Setup
# ============================================================
st.title('🧠 Face Recognition System with ArcFace Embeddings')

# Create data directory for captured images
os.makedirs('data', exist_ok=True)
name_list = os.listdir('data')

# ============================================================
# 1️⃣ Data Collection Stage
# ============================================================
st.sidebar.header("📸 Data Collection")
webcam_channel = st.sidebar.selectbox(
    'Webcam Channel:',
    ('Select Channel', '0', '1', '2', '3')
)
name_person = st.text_input('Enter the Person\'s Name:')
img_number = st.number_input('Number of Images to Capture:', min_value=10, value=50)
FRAME_WINDOW = st.image([])

if webcam_channel != 'Select Channel':
    take_img = st.button('Start Capturing Images')

    if take_img:
        # Prevent duplicate name folders
        if name_person in name_list:
            st.warning('⚠️ The name already exists. Please use a different name.')
        else:
            os.mkdir(f'data/{name_person}')
            st.success(f'✅ Directory created for {name_person}')

            face_classifier = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            cap = cv2.VideoCapture(int(webcam_channel))
            count = 0

            while True:
                success, img = cap.read()
                if not success:
                    st.error('❌ [INFO] Camera not detected!')
                    break

                # Save each frame as image
                cv2.imwrite(f'data/{name_person}/{count}.jpg', img)
                st.info(f'[INFO] Saved {count}.jpg')
                count += 1

                # Draw rectangles around detected faces
                faces = face_classifier.detectMultiScale(img)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                FRAME_WINDOW.image(img, channels='BGR')

                if count >= img_number:
                    st.success(f'[INFO] Collected {img_number} images successfully.')
                    break

            FRAME_WINDOW.image([])
            cap.release()
            cv2.destroyAllWindows()

else:
    st.warning('[INFO] Please select a camera channel to begin.')

# ============================================================
# 2️⃣ Data Normalization Stage
# ============================================================
st.sidebar.header('🧩 Normalize Image Data')
if st.sidebar.button('Normalize'):
    path_to_dir = "data"
    path_to_save = "norm_data"

    flag = True
    detector = MTCNN()

    # Check if normalization has already been done
    class_list_update = []
    class_list_dir = os.listdir(path_to_dir)
    if os.path.exists(path_to_save):
        class_list_save = os.listdir(path_to_save)
        class_list_update = list(set(class_list_dir) ^ set(class_list_save))
    else:
        os.makedirs(path_to_save)

    # Determine which classes need normalization
    if len(class_list_update) == 0:
        if (set(class_list_dir) == set(class_list_save)):
            flag = False
        else:
            class_list = os.listdir(path_to_dir)
    else:
        class_list = class_list_update

    # Run normalization if needed
    if flag:
        class_list = sorted(class_list)
        for name in class_list:
            st.info(f"[INFO] Normalizing class '{name}' ...")
            img_list = glob.glob(os.path.join(path_to_dir, name) + '/*')

            # Create save folder
            save_folder = os.path.join(path_to_save, name)
            os.makedirs(save_folder, exist_ok=True)

            for img_path in img_list:
                img = cv2.imread(img_path)
                detections = detector.detect_faces(img)

                if len(detections) > 0:
                    right_eye = detections[0]['keypoints']['right_eye']
                    left_eye = detections[0]['keypoints']['left_eye']
                    bbox = detections[0]['box']
                    norm_img_roi = alignment_procedure(img, left_eye, right_eye, bbox)
                    cv2.imwrite(f'{save_folder}/{os.path.split(img_path)[1]}', norm_img_roi)
                else:
                    st.warning(f'[INFO] Eyes not detected in {img_path}')

            st.success(f"[INFO] Normalized images for '{name}' saved in '{path_to_save}'")

        st.success('[INFO] All images normalized successfully.')

    else:
        st.warning('[INFO] All data already normalized.')

# ============================================================
# 3️⃣ Model Training Stage
# ============================================================
st.sidebar.header('⚙️ Train Face Recognition Model')
if st.sidebar.button('Train Model'):
    path_to_dir = "norm_data"
    path_to_save = "model.h5"

    # Load ArcFace model for embedding generation
    model = ArcFace.loadModel()
    target_size = model.input_shape[1:3]

    x, y = [], []
    names = sorted([
        n for n in os.listdir(path_to_dir)
        if os.path.isdir(os.path.join(path_to_dir, n)) and len(os.listdir(os.path.join(path_to_dir, n))) > 0
    ])
    class_number = len(names)

    # Generate embeddings for each image
    for name in names:
        st.info(f"[INFO] Generating embeddings for class '{name}' ...")
        img_list = sorted(glob.glob(os.path.join(path_to_dir, name) + '/*'))
        for img_path in img_list:
            img = cv2.imread(img_path)
            img_resize = cv2.resize(img, target_size)
            img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_norm = img_pixels / 255
            img_embedding = model.predict(img_norm)[0]
            x.append(img_embedding)
            y.append(name)
        st.success(f"[INFO] Completed embeddings for '{name}'")

    st.success('[INFO] All embeddings generated successfully.')

    # Prepare training data
    df = pd.DataFrame(x, columns=np.arange(512))
    df['names'] = y
    x = df.copy()
    y = x.pop('names')
    y, _ = y.factorize()
    x = x.astype('float64')
    y = keras.utils.to_categorical(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Build recognition classifier
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

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        path_to_save,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=200)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

    # Start model training
    st.success('[INFO] Training model ...')
    history = model.fit(
        x_train, y_train,
        epochs=800,
        batch_size=16,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint, earlystopping, lr_scheduler]
    )
    st.success('[INFO] Training completed successfully!')

    # Save performance plots
    metric_loss, val_loss = history.history['loss'], history.history['val_loss']
    acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
    epochs = range(len(metric_loss))

    # Loss Plot
    plt.figure()
    plt.plot(epochs, metric_loss, 'blue', label='Loss')
    plt.plot(epochs, val_loss, 'red', label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.png', bbox_inches='tight')
    st.image('loss.png', caption='Training vs Validation Loss')

    # Accuracy Plot
    plt.figure()
    plt.plot(epochs, acc, 'green', label='Accuracy')
    plt.plot(epochs, val_acc, 'orange', label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy.png', bbox_inches='tight')
    st.image('accuracy.png', caption='Training vs Validation Accuracy')

# ============================================================
# 4️⃣ Inference Stage
# ============================================================
st.sidebar.header('🎯 Inference (Real-Time Recognition)')
threshold = st.sidebar.slider('Confidence Threshold', 0.01, 0.99, 0.6)

if st.sidebar.button('Run/Stop Inference'):
    class_names = sorted(os.listdir('data'))

    if webcam_channel != 'Select Channel':
        cap = cv2.VideoCapture(int(webcam_channel))
        detector = MTCNN()
        arcface_model = ArcFace.loadModel()
        target_size = arcface_model.input_shape[1:3]
        face_rec_model = load_model('model.h5', compile=True)

        while True:
            success, img = cap.read()
            if not success:
                st.warning('[INFO] Camera not responding.')
                break

            detections = detector.detect_faces(img)
            if len(detections) > 0:
                for detect in detections:
                    right_eye = detect['keypoints']['right_eye']
                    left_eye = detect['keypoints']['left_eye']
                    bbox = detect['box']
                    xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])

                    norm_img_roi = alignment_procedure(img, left_eye, right_eye, bbox)
                    img_resize = cv2.resize(norm_img_roi, target_size)
                    img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_norm = img_pixels / 255
                    img_embedding = arcface_model.predict(img_norm)[0]

                    data = pd.DataFrame([img_embedding], columns=np.arange(512))
                    predict = face_rec_model.predict(data)[0]

                    if max(predict) > threshold:
                        pose_class = class_names[predict.argmax()]
                    else:
                        pose_class = 'Unknown Person'

                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(img, f'{pose_class}', (xmin, ymin-10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            else:
                st.warning('[INFO] No faces detected in this frame.')

            FRAME_WINDOW.image(img, channels='BGR')

        FRAME_WINDOW.image([])
        st.success('[INFO] Inference completed.')

    else:
        st.warning('[INFO] Please select a camera channel first.')
