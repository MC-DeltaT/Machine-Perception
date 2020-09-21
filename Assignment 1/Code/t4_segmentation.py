# Performs k-means segmentation on the sample images, using various feature descriptors.

import cv2 as cv
import numpy
import os.path
from pathlib import Path

INPUTS = ['data/card.png', 'data/dugong.jpg']
OUTPUT_DIR = 'results/{}'
OUTPUT_FILE = '{}-segmentation.png'
PERFORMANCE_FILE = 'segmentation_performance.txt'
KMEANS_CLUSTERS = 3
KMEANS_ITERATIONS = 100
KMEANS_EPSILON = 0.1
KMEANS_ATTEMPTS = 10
SEGMENT_COLOURS = numpy.array([
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255)
])

for file in INPUTS:
    image = cv.imread(file, cv.IMREAD_COLOR)
    filename = os.path.splitext(os.path.basename(file))[0]
    output_dir = OUTPUT_DIR.format(filename)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_dir, PERFORMANCE_FILE), 'w') as performance_file:
        features = [
            ('greyscale', cv.cvtColor(image, cv.COLOR_BGR2GRAY)[:, :, numpy.newaxis]),
            ('rgb', image),
            ('hsv', cv.cvtColor(image, cv.COLOR_BGR2HSV)),
            ('hls', cv.cvtColor(image, cv.COLOR_BGR2HLS)),
            ('xyz', cv.cvtColor(image, cv.COLOR_BGR2XYZ)),
            ('lab', cv.cvtColor(image, cv.COLOR_BGR2LAB)),
            ('luv', cv.cvtColor(image, cv.COLOR_BGR2LUV)),
            ('yuv', cv.cvtColor(image, cv.COLOR_BGR2YUV))
        ]

        for method_name, data in features:
            height, width, feature_size = data.shape
            data = data.reshape((width * height, feature_size)).astype(numpy.float32)
            data = cv.normalize(data, None, 0, 1, cv.NORM_MINMAX)
            kmeans_criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, KMEANS_ITERATIONS, KMEANS_EPSILON)
            kmeans_flags = cv.KMEANS_RANDOM_CENTERS
            compactness, best_labels, centres = cv.kmeans(data, KMEANS_CLUSTERS, None, kmeans_criteria, KMEANS_ATTEMPTS, kmeans_flags)
            best_labels = best_labels.reshape((height, width))
            result = SEGMENT_COLOURS[best_labels]
            output_file = os.path.join(output_dir, OUTPUT_FILE.format(method_name))
            cv.imwrite(output_file, result)
            performance_file.write(f'{method_name}: {round(compactness)}\n')
