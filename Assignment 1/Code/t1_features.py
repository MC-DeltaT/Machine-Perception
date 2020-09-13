from glob import glob
import os.path
import cv2 as cv
from matplotlib import pyplot
import numpy

INPUT = [
    'data/card.png',
    'data/dugong.jpg',
    'results/card-scaled_*x.png',
    'results/card-rotated_*deg.png',
    'results/dugong-scaled_*x.png',
    'results/dugong-rotated_*deg.png',
]
HISTOGRAM_OUTPUT_FILE = 'results/{}-hist.png'
HARRIS_OUTPUT_FILE = 'results/{}-harris.png'
SIFT_KEYPOINTS_OUTPUT_FILE = 'results/{}-keypoints.png'
HISTOGRAM_BINS = 16
HARRIS_BLOCK_SIZE = 3
HARRIS_SOBEL_SIZE = 3
HARRIS_K = 0.05
HARRIS_THRESHOLD = 0.15

for file in [f for p in INPUT for f in glob(p)]:
    image = cv.imread(file, cv.IMREAD_COLOR)
    image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    filename = os.path.splitext(os.path.basename(file))[0]

    pyplot.clf()
    pyplot.hist(image_grey.flatten(), HISTOGRAM_BINS, (0, 255))
    pyplot.xlabel('Intensity')
    pyplot.ylabel('Frequency')
    pyplot.ylim((0, image_grey.shape[0] * image_grey.shape[1]))
    pyplot.tight_layout()
    output_file = HISTOGRAM_OUTPUT_FILE.format(filename)
    pyplot.savefig(output_file)

    harris_response = cv.cornerHarris(image_grey, HARRIS_BLOCK_SIZE, HARRIS_SOBEL_SIZE, HARRIS_K)
    corners = harris_response > HARRIS_THRESHOLD * harris_response.max()
    result = image.copy()
    for (i, j) in zip(*numpy.where(corners)):
        result = cv.drawMarker(result, (j, i), (0, 0, 0), markerSize=15, thickness=1)
    output_file = HARRIS_OUTPUT_FILE.format(filename)
    cv.imwrite(output_file, result)
