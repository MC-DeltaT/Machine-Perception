# Visualises Harris corners in the scaled and rotated images.
# gen_scaled.py and gen_rotated.py should be run before running this script.

from glob import glob
import os.path
from pathlib import Path
import cv2 as cv
import numpy

BASE_IMAGES = [('card', 0.005), ('dugong', 0.0003)]
TRANSFORMS = ['scaled', 'rotated']
INPUT_FILES = 'generated/{}/{}/*.png'
OUTPUT_DIR = 'results/{}/{}'
OUTPUT_FILE = '{}-harris.png'
HARRIS_BLOCK_SIZE = 3
HARRIS_SOBEL_SIZE = 3
HARRIS_K = 0.05

for base_image, harris_threshold in BASE_IMAGES:
    for transform in TRANSFORMS:
        file_pattern = INPUT_FILES.format(base_image, transform)
        output_dir = OUTPUT_DIR.format(base_image, transform)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for file in glob(file_pattern):
            image = cv.imread(file, cv.IMREAD_COLOR)
            image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            filename = os.path.splitext(os.path.basename(file))[0]

            harris_response = cv.cornerHarris(image_grey, HARRIS_BLOCK_SIZE, HARRIS_SOBEL_SIZE, HARRIS_K)
            corners = harris_response > harris_threshold
            result = image.copy()
            for (i, j) in zip(*numpy.where(corners)):
                marker_size = min(image_grey.shape[0], image_grey.shape[1]) // 10
                result = cv.drawMarker(result, (j, i), (0, 0, 0), markerSize=marker_size, thickness=1)
            output_file = os.path.join(output_dir, OUTPUT_FILE.format(filename))
            cv.imwrite(output_file, result)
