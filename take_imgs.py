import os
import cv2
import argparse

# ============================================
# Argument Parser: Define command-line options
# ============================================
ap = argparse.ArgumentParser()

# RTSP link or webcam ID (e.g., "0" for default webcam)
ap.add_argument("-i", "--source", type=str, required=True,
                help="RTSP link or webcam-id")

# Name of the person whose images are being collected
ap.add_argument("-n", "--name", type=str, required=True,
                help="Name of the person")

# Directory where the collected images will be saved
ap.add_argument("-o", "--save", type=str, default='Data',
                help="Path to save directory")

# Minimum confidence threshold for face detection (0 < conf < 1)
ap.add_argument("-c", "--conf", type=float, default=0.5,
                help="Minimum prediction confidence (0 < conf < 1)")

# Number of images to collect
ap.add_argument("-x", "--number", type=int, default=100,
                help="Number of images to collect")

# Parse the command-line arguments
args = vars(ap.parse_args())
source = args["source"]
name_of_person = args['name']
path_to_save = args['save']
min_confidence = args["conf"]

# ============================================
# Create the output directory (if not exists)
# ============================================
os.makedirs((os.path.join(path_to_save, name_of_person)), exist_ok=True)
path_to_save = os.path.join(path_to_save, name_of_person)

# ============================================
# Load the pre-trained Caffe face detection model
# ============================================
# - deploy.prototxt: defines the model architecture
# - .caffemodel: contains the pre-trained weights
opencv_dnn_model = cv2.dnn.readNetFromCaffe(
    prototxt="models/deploy.prototxt",
    caffeModel="models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# If input source is numeric (e.g., "0"), convert to int for webcam access
if source.isnumeric():
    source = int(source)

# Initialize video capture (from webcam or RTSP stream)
cap = cv2.VideoCapture(source)
# Get the video frame rate (FPS) to determine capture frequency
fps = cap.get(cv2.CAP_PROP_FPS)

count = 0  # Frame counter
while True:
    success, img = cap.read()
    if not success:
        print('[INFO] Camera not working or video stream ended!')
        break

    # ============================================
    # Save an image every (FPS / 5) frames
    # e.g., if FPS = 30, saves approximately every 0.2s
    # ============================================
    if count % int(fps / 5) == 0:
        img_name = len(os.listdir(path_to_save))
        cv2.imwrite(f'{path_to_save}/{img_name}.jpg', img)
        print(f'[INFO] Successfully saved {img_name}.jpg')
    count += 1

    # ============================================
    # Run face detection using OpenCV DNN
    # ============================================
    h, w, _ = img.shape
    preprocessed_image = cv2.dnn.blobFromImage(
        img, scalefactor=1.0, size=(300, 300),
        mean=(104.0, 117.0, 123.0), swapRB=False, crop=False
    )
    opencv_dnn_model.setInput(preprocessed_image)
    results = opencv_dnn_model.forward()

    # ============================================
    # Draw bounding boxes around detected faces
    # ============================================
    for face in results[0][0]:
        face_confidence = face[2]
        if face_confidence > min_confidence:
            bbox = face[3:]
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)

            cv2.rectangle(
                img, pt1=(x1, y1), pt2=(x2, y2),
                color=(0, 255, 0), thickness=w // 200
            )

    # ============================================
    # Display the video stream in a window
    # ============================================
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

    # ============================================
    # Stop collecting after reaching the desired number of images
    # ============================================
    if img_name == args["number"] - 1:
        print(f"[INFO] Collected {args['number']} images.")
        cv2.destroyAllWindows()
        break
