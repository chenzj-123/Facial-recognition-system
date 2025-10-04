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
# Argument Parser — Define input arguments for command line
# ============================================================
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="Path to the input image for testing")
ap.add_argument("-m", "--model", type=str, default='models/model.h5',
                help="Path to trained face recognition model (.h5)")
ap.add_argument("-c", "--conf", type=float, default=0.9,
                help="Minimum prediction confidence (0 < conf < 1)")

# Liveness Model
ap.add_argument("-lm", "--liveness_model", type=str, default='models/liveness.model',
                help="Path to the trained liveness detection model")
ap.add_argument("-le", "--label_encoder", type=str, default='models/le.pickle',
                help="Path to the label encoder (.pickle) file")

args = vars(ap.parse_args())
path_to_img = args["image"]
path_saved_model = args["model"]
threshold = args["conf"]

# ============================================================
# Load trained models
# ============================================================
# Load the face recognition classifier model
face_rec_model = load_model(path_saved_model, compile=True)

# Initialize MTCNN for face detection
detector = MTCNN()

# Load ArcFace model to generate 512-dimensional embeddings
arcface_model = ArcFace.loadModel()
target_size = arcface_model.layers[0].input_shape[0][1:3]

# Load liveness detection model and label encoder
liveness_model = tf.keras.models.load_model(args['liveness_model'])
label_encoder = pickle.loads(open(args["label_encoder"], "rb").read())

# ============================================================
# Read the input image and detect faces using MTCNN
# ============================================================
img = cv2.imread(path_to_img)
detections = detector.detect_faces(img)

# If at least one face is detected
if len(detections) > 0:
    for detect in detections:

        # Extract bounding box coordinates
        bbox = detect['box']
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), \
            int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])

        # ============================================================
        # Liveness Detection Stage
        # ============================================================
        img_roi = img[ymin:ymax, xmin:xmax]                # Crop face region
        face_resize = cv2.resize(img_roi, (32, 32))        # Resize for liveness model
        face_norm = face_resize.astype("float") / 255.0    # Normalize pixel values to [0,1]
        face_array = tf.keras.preprocessing.image.img_to_array(face_norm)
        face_prepro = np.expand_dims(face_array, axis=0)   # Add batch dimension

        preds_liveness = liveness_model.predict(face_prepro)[0]
        decision = np.argmax(preds_liveness)               # 0 = Fake, 1 = Real

        # ============================================================
        # If the face is FAKE (spoofing detected)
        # ============================================================
        if decision == 0:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(
                img, 'Fake',
                (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN,
                2, (0, 0, 255), 2
            )

        # ============================================================
        # If the face is REAL → Proceed to Recognition
        # ============================================================
        else:
            # Extract eyes and align the face using custom alignment function
            right_eye = detect['keypoints']['right_eye']
            left_eye = detect['keypoints']['left_eye']
            norm_img_roi = alignment_procedure(img, left_eye, right_eye, bbox)

            # Resize to ArcFace input size
            img_resize = cv2.resize(norm_img_roi, target_size)

            # ------------------------------------------------------------
            # Preprocess face for ArcFace embedding generation
            # ------------------------------------------------------------
            # Converts the image to array, adds batch dimension,
            # and normalizes to [0, 1] before feeding into ArcFace.
            img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_norm = img_pixels / 255.0
            img_embedding = arcface_model.predict(img_norm)[0]

            # Convert embedding into a DataFrame (same format used in training)
            data = pd.DataFrame([img_embedding], columns=np.arange(512))

            # Predict using the trained classifier
            predict = face_rec_model.predict(data)[0]

            # ------------------------------------------------------------
            # Determine if confidence passes the recognition threshold
            # ------------------------------------------------------------
            if max(predict) > threshold:
                class_id = predict.argmax()
                pose_class = label_encoder.classes_[class_id]
                color = (0, 255, 0)   # Green for known faces
            else:
                pose_class = 'Unknown Person'
                color = (0, 0, 255)   # Red for unknown faces

            # ------------------------------------------------------------
            # Display recognition result on image
            # ------------------------------------------------------------
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(
                img, f'{pose_class}',
                (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN,
                2, (255, 0, 255), 2
            )

    # ============================================================
    # Display final annotated image
    # ============================================================
    cv2.imshow('Image', img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
