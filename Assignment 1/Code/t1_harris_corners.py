# Visualises Harris corners in the scaled and rotated images.

from glob import glob
import os.path
import cv2 as cv
import numpy

INPUTS = [
    ([
         'data/card.png',
         'results/card-scaled_*x.png',
         'results/card-rotated_*deg.png'
     ], 0.005),
    ([
         'data/dugong.jpg',
         'results/dugong-scaled_*x.png',
         'results/dugong-rotated_*deg.png',
     ], 0.0003)
]
OUTPUT_FILE = 'results/{}-harris.png'
HARRIS_BLOCK_SIZE = 3
HARRIS_SOBEL_SIZE = 3
HARRIS_K = 0.05

for file_patterns, harris_threshold in INPUTS:
    for file in [f for p in file_patterns for f in glob(p)]:
        image = cv.imread(file, cv.IMREAD_COLOR)
        image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        filename = os.path.splitext(os.path.basename(file))[0]

        harris_response = cv.cornerHarris(image_grey, HARRIS_BLOCK_SIZE, HARRIS_SOBEL_SIZE, HARRIS_K)
        corners = harris_response > harris_threshold
        result = image.copy()
        for (i, j) in zip(*numpy.where(corners)):
            marker_size = min(image_grey.shape[0], image_grey.shape[1]) // 10
            result = cv.drawMarker(result, (j, i), (0, 0, 0), markerSize=marker_size, thickness=1)
        output_file = OUTPUT_FILE.format(filename)
        cv.imwrite(output_file, result)
