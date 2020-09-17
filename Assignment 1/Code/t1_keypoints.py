# Visualises keypoint locations in the scaled and rotated images.
# gen_scaled.py and gen_rotated.py should be run before running this script.

from glob import glob
import os.path
from pathlib import Path
import cv2 as cv

BASE_IMAGES = ['card', 'dugong']
TRANSFORMS = ['scaled', 'rotated']
INPUT_FILES = 'generated/{}/{}/*.png'
OUTPUT_DIR = 'results/{}/{}'
OUTPUT_FILE = '{}-keypoints.png'
SCALE = 2

for base_image in BASE_IMAGES:
    for transform in TRANSFORMS:
        file_pattern = INPUT_FILES.format(base_image, transform)
        output_dir = OUTPUT_DIR.format(base_image, transform)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for file in glob(file_pattern):
            image = cv.imread(file, cv.IMREAD_COLOR)
            filename = os.path.splitext(os.path.basename(file))[0]

            sift = cv.xfeatures2d.SIFT_create()
            keypoints = sift.detect(image, None)
            # Scale everything up so we can see it better (doesn't affect keypoint location)
            for kp in keypoints:
                kp.pt = (kp.pt[0] * SCALE, kp.pt[1] * SCALE)
                kp.size *= SCALE
            result = cv.resize(image, (0, 0), fx=SCALE, fy=SCALE)
            result = cv.drawKeypoints(result, keypoints, None)
            output_file = os.path.join(output_dir, OUTPUT_FILE.format(filename))
            cv.imwrite(output_file, result)
