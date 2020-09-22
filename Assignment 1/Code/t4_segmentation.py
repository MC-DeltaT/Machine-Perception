# Performs k-means segmentation on the sample images, using various feature descriptors.

import cv2 as cv
import numpy
import os.path
from pathlib import Path

INPUTS = ['data/card.png', 'data/dugong.jpg']
OUTPUT_DIR = 'results/{}/segmentation'
OUTPUT_FILE = '{}.png'
PERFORMANCE_FILE = 'performance.txt'
KMEANS_CLUSTERS = 3
KMEANS_ITERATIONS = 200
KMEANS_EPSILON = 0.01
KMEANS_ATTEMPTS = 5
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

    features = [
        ('greyscale', cv.cvtColor(image, cv.COLOR_BGR2GRAY)[:, :, numpy.newaxis]),
        ('rgb', image),
        ('hsv', cv.cvtColor(image, cv.COLOR_BGR2HSV)),
        ('hls', cv.cvtColor(image, cv.COLOR_BGR2HLS)),
        ('xyz', cv.cvtColor(image, cv.COLOR_BGR2XYZ)),
        ('lab', cv.cvtColor(image, cv.COLOR_BGR2LAB)),
        ('luv', cv.cvtColor(image, cv.COLOR_BGR2LUV)),
        ('yuv', cv.cvtColor(image, cv.COLOR_BGR2YUV)),
        ('red', image[:, :, 2][:, :, numpy.newaxis]),
        ('green', image[:, :, 1][:, :, numpy.newaxis]),
        ('blue', image[:, :, 0][:, :, numpy.newaxis]),
        ('hue', cv.cvtColor(image, cv.COLOR_BGR2HSV)[:, :, 0][:, :, numpy.newaxis]),
        ('saturation', cv.cvtColor(image, cv.COLOR_BGR2HSV)[:, :, 1][:, :, numpy.newaxis])
    ]

    method_performance = []
    for method_name, data in features:
        height, width, feature_size = data.shape
        data = data.reshape((width * height, feature_size)).astype(numpy.float32)
        data = (data - numpy.min(data)) / (numpy.max(data) - numpy.min(data))
        kmeans_criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, KMEANS_ITERATIONS, KMEANS_EPSILON)
        kmeans_flags = cv.KMEANS_RANDOM_CENTERS
        compactness, best_labels, centres = cv.kmeans(data, KMEANS_CLUSTERS, None, kmeans_criteria, KMEANS_ATTEMPTS, kmeans_flags)
        best_labels = best_labels.reshape((height, width))
        result = SEGMENT_COLOURS[best_labels]
        output_file = os.path.join(output_dir, OUTPUT_FILE.format(method_name))
        cv.imwrite(output_file, result)
        method_performance.append((method_name, compactness))

    method_performance = sorted(method_performance, key=lambda t: t[1])
    with open(os.path.join(output_dir, PERFORMANCE_FILE), 'w') as performance_file:
        for method, compactness in method_performance:
            performance_file.write(f'{method}: {round(compactness)}\n')
