import os
import cv2
import glob
import argparse
from my_utils import alignment_procedure
from mtcnn import MTCNN

# ============================================================
# Argument Parser — Define command-line parameters
# ============================================================
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="Path to the input dataset directory")
ap.add_argument("-o", "--save", type=str, default='Norm',
                help="Path to directory for saving normalized images")
args = vars(ap.parse_args())

path_to_dir = args["dataset"]
path_to_save = args["save"]

# ============================================================
# Initialize MTCNN Detector and Control Variables
# ============================================================
Flag = True   # Used to determine whether normalization is required
detector = MTCNN()   # Multi-task CNN for face detection and landmark extraction

# ============================================================
# Determine Which Classes Need Normalization
# ============================================================
# The code checks if the output folder already contains normalized data.
# It only normalizes missing classes or newly added data.

class_list_update = []   # New classes that need normalization
class_list_dir = []      # Classes found in the raw dataset
class_list_save = []     # Classes already normalized

if os.path.exists(path_to_save):
    class_list_save = os.listdir(path_to_save)
    class_list_dir = os.listdir(path_to_dir)
    # Compute the symmetric difference between folders (new or missing)
    class_list_update = list(set(class_list_dir) ^ set(class_list_save))
else:
    os.makedirs(path_to_save)

# Determine which classes to process
if len(class_list_update) == 0:
    if len(class_list_dir) == 0 and len(class_list_save) == 0:
        class_list = os.listdir(path_to_dir)
    else:
        # If both directories contain the same class names, skip normalization
        if set(class_list_dir) == set(class_list_save):
            Flag = False
        else:
            Flag = True
else:
    class_list = class_list_update

# ============================================================
# Normalize Images (Align Faces Using Eye Coordinates)
# ============================================================
if Flag:
    class_list = sorted(class_list)
    for name in class_list:
        img_list = glob.glob(os.path.join(path_to_dir, name) + '/*')

        # Create subdirectory for normalized images
        save_folder = os.path.join(path_to_save, name)
        os.makedirs(save_folder, exist_ok=True)

        for img_path in img_list:
            img = cv2.imread(img_path)

            # Run face detection with MTCNN
            detections = detector.detect_faces(img)

            # If at least one face is detected
            if len(detections) > 0:
                right_eye = detections[0]['keypoints']['right_eye']
                left_eye = detections[0]['keypoints']['left_eye']
                bbox = detections[0]['box']

                # Align face using custom alignment procedure
                norm_img_roi = alignment_procedure(img, left_eye, right_eye, bbox)

                # Save normalized (aligned + cropped) image
                cv2.imwrite(f'{save_folder}/{os.path.split(img_path)[1]}', norm_img_roi)
                print(f'[INFO] Successfully normalized {img_path}')

            else:
                print(f'[INFO] No eyes detected in {img_path}, skipping.')

        print(f'[INFO] Completed normalization for class "{name}" '
              f'({len(os.listdir(save_folder))} images)')
    print(f"[INFO] All normalized images saved in '{path_to_save}'")

# ============================================================
# If All Data Already Normalized
# ============================================================
else:
    print('[INFO] All data already normalized. Nothing to update.')
