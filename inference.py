from keras.models import load_model
from mtcnn import MTCNN
from my_utils import alignment_procedure
import tensorflow as tf
import ArcFace
import cv2
import numpy as np
import pandas as pd
import argparse
import pickle

# ============================================================
# Argument Parser — Define CLI arguments for flexibility
# ============================================================
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--source", type=str, required=True,
                help="Path to video file or webcam index (e.g., 0 for default webcam)")
ap.add_argument("-m", "--model", type=str, default='models/model.h5',
                help="Path to trained .h5 face recognition model")
ap.add_argument("-c", "--conf", type=float, default=0.9,
                help="Minimum prediction confidence threshold (0 < conf < 1)")

# Liveness detection model arguments
ap.add_argument("-lm", "--liveness_model", type=str, default='models/liveness.model',
                help="Path to trained liveness detection model")
ap.add_argument("-le", "--label_encoder", type=str, default='models/le.pickle',
                help="Path to label encoder for recognition model")

args = vars(ap.parse_args())
source = args["source"]
path_saved_model = args["model"]
threshold = args["conf"]

# Convert webcam index string to int (for OpenCV video capture)
if source.isnumeric():
    source = int(source)

# ============================================================
# Load All Required Models
# ============================================================
print("[INFO] Loading models...")

# Face recognition classifier (trained on ArcFace embeddings)
face_rec_model = load_model(path_saved_model, compile=True)

# MTCNN face detector for bounding boxes and keypoints
detector = MTCNN()

# ArcFace model to generate 512-dimensional embeddings from aligned faces
arcface_model = ArcFace.loadModel()
target_size = arcface_model.layers[0].input_shape[0][1:3]

# Liveness detection model (binary: real vs fake)
liveness_model = tf.keras.models.load_model(args['liveness_model'])

# Label encoder for mapping prediction indices to names
label_encoder = pickle.loads(open(args["label_encoder"], "rb").read())

# Optionally load class names for displaying results
try:
    with open('models/class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
except FileNotFoundError:
    print("[WARNING] Class names file not found. Defaulting to numeric labels.")
    class_names = [str(i) for i in range(face_rec_model.output_shape[-1])]

# ============================================================
# Initialize Video Stream
# ============================================================
cap = cv2.VideoCapture(source)
print("[INFO] Starting video stream... Press 'q' to quit.")

# ============================================================
# Process Each Frame from Video / Webcam
# ============================================================
while True:
    success, img = cap.read()
    if not success:
        print('[INFO] Camera error or video stream ended.')
        break

    # Detect faces using MTCNN
    detections = detector.detect_faces(img)

    if len(detections) > 0:
        for detect in detections:
            bbox = detect['box']
            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), \
                int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])

            # ============================================================
            # Step 1: Liveness Detection
            # ============================================================
            # Crop and preprocess face region
            img_roi = img[ymin:ymax, xmin:xmax]
            face_resize = cv2.resize(img_roi, (32, 32))
            face_norm = face_resize.astype("float") / 255.0
            face_array = tf.keras.preprocessing.image.img_to_array(face_norm)
            face_prepro = np.expand_dims(face_array, axis=0)

            # Predict whether the face is real (1) or fake (0)
            preds_liveness = liveness_model.predict(face_prepro)[0]
            decision = np.argmax(preds_liveness)

            # ============================================================
            # Step 2: If Fake Face → Mark as Spoofing
            # ============================================================
            if decision == 0:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.putText(
                    img, 'Fake',
                    (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 255), 2
                )

            # ============================================================
            # Step 3: If Real Face → Perform Recognition
            # ============================================================
            else:
                # Extract eyes for geometric alignment
                right_eye = detect['keypoints']['right_eye']
                left_eye = detect['keypoints']['left_eye']

                # Align face using custom alignment procedure
                norm_img_roi = alignment_procedure(img, left_eye, right_eye, bbox)

                # Resize to match ArcFace input dimensions
                img_resize = cv2.resize(norm_img_roi, target_size)

                # Convert to Keras-compatible tensor and normalize
                img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_norm = img_pixels / 255.0

                # Generate ArcFace embedding (512-D vector)
                img_embedding = arcface_model.predict(img_norm)[0]

                # Pass embedding to classifier for identity prediction
                data = pd.DataFrame([img_embedding], columns=np.arange(512))
                predict = face_rec_model.predict(data)[0]

                # Get highest confidence score and class index
                confidence = max(predict)
                class_index = predict.argmax()

                # Assign label based on confidence threshold
                if confidence > threshold:
                    pose_class = class_names[class_index]
                    box_color = (0, 255, 0)  # Green = recognized
                else:
                    pose_class = 'Unknown Person'
                    box_color = (0, 0, 255)  # Red = unknown

                # ============================================================
                # Step 4: Display Results
                # ============================================================
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), box_color, 2)
                cv2.putText(
                    img,
                    f'{pose_class} ({confidence:.2f})',
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 0, 255),
                    2
                )

    else:
        print('[INFO] No faces detected in current frame.')

    # Show live video with annotations
    cv2.imshow('Output Image', img)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

print('[INFO] Video inference ended successfully.')
