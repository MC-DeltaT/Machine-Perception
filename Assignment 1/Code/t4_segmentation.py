# Performs k-means segmentation on the sample images, using various feature descriptors.
# For external validation, additional human-labelled images are supplied.

import cv2 as cv
from itertools import permutations
import numpy
import os.path
from pathlib import Path

INPUTS = [
    ('data/card.png', 'data/card-segment_labels.png'),
    ('data/dugong.jpg', 'data/dugong-segment_labels.png')
]
OUTPUT_DIR = 'results/{}/segmentation'
OUTPUT_FILE = '{}.png'
PERFORMANCE_FILE = 'performance.txt'
CLUSTERS = 3
ITERATIONS = 200
EPSILON = 0.01
ATTEMPTS = 5
CLUSTER_COLOURS = numpy.array([
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255)
])

#label_permutations = numpy.array(list(permutations(range(KMEANS_CLUSTERS))))
for image_file, labels_file in INPUTS:
    image = cv.imread(image_file, cv.IMREAD_COLOR)
    filename = os.path.splitext(os.path.basename(image_file))[0]
    output_dir = OUTPUT_DIR.format(filename)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    correct_labels = cv.imread(labels_file, cv.IMREAD_GRAYSCALE).ravel()

    features = [
        ('intensity', cv.cvtColor(image, cv.COLOR_BGR2GRAY)[:, :, numpy.newaxis]),
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

    feature_performances = []
    for feature_name, data in features:
        height, width, feature_size = data.shape
        data = data.reshape((width * height, feature_size)).astype(numpy.float32)
        data = (data - numpy.min(data)) / (numpy.max(data) - numpy.min(data))
        kmeans_criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, ITERATIONS, EPSILON)
        kmeans_flags = cv.KMEANS_RANDOM_CENTERS
        ssdc, labels, centres = cv.kmeans(data, CLUSTERS, None, kmeans_criteria, ATTEMPTS, kmeans_flags)
        labels = labels.reshape((height, width))
        result = CLUSTER_COLOURS[labels]
        output_file = os.path.join(output_dir, OUTPUT_FILE.format(feature_name))
        cv.imwrite(output_file, result)

        labels = labels.ravel()
        confusion_matrix = numpy.array([numpy.bincount(labels[correct_labels == l], minlength=CLUSTERS)
                                        for l in range(CLUSTERS)])
        total = labels.shape[0] / 2 * (labels.shape[0] - 1)
        points_per_class = numpy.sum(confusion_matrix, axis=1)
        points_per_cluster = numpy.sum(confusion_matrix, axis=0)
        tp_fp = numpy.sum(points_per_cluster / 2 * (points_per_cluster - 1))
        tp_fn = numpy.sum(points_per_class / 2 * (points_per_class - 1))
        tp = numpy.sum(confusion_matrix / 2 * (confusion_matrix - 1))
        tn = total - tp_fp - tp_fn + tp
        rand_index = (tp + tn) / total
        feature_performances.append((feature_name, ssdc, rand_index))

    feature_performances = sorted(feature_performances, key=lambda t: (t[1], t[2]))
    with open(os.path.join(output_dir, PERFORMANCE_FILE), 'w') as performance_file:
        for method, ssdc, rand_index in feature_performances:
            performance_file.write(f'{method}: {round(ssdc)}, {rand_index:.4f}\n')
