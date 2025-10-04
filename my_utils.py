import numpy as np
import math
from PIL import Image
from deepface.modules import verification


def alignment_procedure(img, left_eye, right_eye, bbox):
    """
    Aligns and crops a detected face based on the coordinates of the left and right eyes.

    Parameters
    ----------
    img : np.ndarray
        Original image containing the detected face.
    left_eye : tuple (x, y)
        Coordinates of the left eye.
    right_eye : tuple (x, y)
        Coordinates of the right eye.
    bbox : list or tuple (x, y, w, h)
        Bounding box around the detected face.

    Returns
    -------
    np.ndarray
        Cropped and rotated (aligned) face region.
    """

    # ------------------------------------------------------
    # 1. Crop the face region using the bounding box
    # ------------------------------------------------------
    x, y, w, h = bbox
    img_roi = img[y:y + h, x:x + w]

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # ------------------------------------------------------
    # 2. Determine rotation direction based on eye positions
    # ------------------------------------------------------
    # If the left eye is lower than the right eye → rotate clockwise
    # If the right eye is lower than the left eye → rotate counterclockwise
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # Clockwise
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1   # Counterclockwise

    # ------------------------------------------------------
    # 3. Compute lengths of the triangle sides (eyes + third point)
    # ------------------------------------------------------
    a = verification.find_euclidean_distance(np.array(left_eye), np.array(point_3rd))
    b = verification.find_euclidean_distance(np.array(right_eye), np.array(point_3rd))
    c = verification.find_euclidean_distance(np.array(right_eye), np.array(left_eye))

    # ------------------------------------------------------
    # 4. Compute rotation angle using the cosine rule
    # ------------------------------------------------------
    # cos(A) = (b² + c² - a²) / (2bc)
    # A is the angle between eyes and horizontal line
    if b != 0 and c != 0:  # Prevent division by zero
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)              # Angle in radians
        angle = (angle * 180) / math.pi       # Convert to degrees

        # Adjust rotation direction
        if direction == -1:
            angle = 90 - angle

        # ------------------------------------------------------
        # 5. Rotate the cropped face (ROI) to align eyes horizontally
        # ------------------------------------------------------
        img_roi = Image.fromarray(img_roi)
        img_roi = np.array(img_roi.rotate(direction * angle))

    # ------------------------------------------------------
    # 6. Return aligned face region
    # ------------------------------------------------------
    return img_roi
