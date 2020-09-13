from glob import glob
import os.path
import cv2 as cv
import numpy

CARD_INPUT = [
    'data/card.png',
    'results/card-scaled_*x.png',
    'results/card-rotated_*deg.png'
]
DUGONG_INPUT = [
    'data/dugong.jpg',
    'results/dugong-scaled_*x.png',
    'results/dugong-rotated_*deg.png',
]
OUTPUT_FILE = 'results/{}-harris.png'
HARRIS_BLOCK_SIZE = 3
HARRIS_SOBEL_SIZE = 3
HARRIS_K = 0.05
CARD_HARRIS_THRESHOLD = 0.005
DUGONG_HARRIS_THRESHOLD = 0.0003


def do_harris_corners(file: str, threshold: float) -> None:
    image = cv.imread(file, cv.IMREAD_COLOR)
    image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    filename = os.path.splitext(os.path.basename(file))[0]

    harris_response = cv.cornerHarris(image_grey, HARRIS_BLOCK_SIZE, HARRIS_SOBEL_SIZE, HARRIS_K)
    corners = harris_response > threshold
    result = image.copy()
    for (i, j) in zip(*numpy.where(corners)):
        marker_size = min(image_grey.shape[0], image_grey.shape[1]) // 10
        result = cv.drawMarker(result, (j, i), (0, 0, 0), markerSize=marker_size, thickness=1)
    output_file = OUTPUT_FILE.format(filename)
    cv.imwrite(output_file, result)


for file in [f for p in CARD_INPUT for f in glob(p)]:
    do_harris_corners(file, CARD_HARRIS_THRESHOLD)
for file in [f for p in DUGONG_INPUT for f in glob(p)]:
    do_harris_corners(file, DUGONG_HARRIS_THRESHOLD)
