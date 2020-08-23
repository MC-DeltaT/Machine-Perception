import glob
import os.path

import cv2 as cv
import numpy


INPUT_FILES = 'data/prac03ex01img*'
OUTPUT_FILE = 'output/{}-{}_corners.png'

MAX_CORNERS = 20
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 10


CORNER_GENERATORS = [
    ('harris', lambda i: cv.goodFeaturesToTrack(i, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE, useHarrisDetector=True)),
    ('shi_tomasi', lambda i: cv.goodFeaturesToTrack(i, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE))
]

for alg_name, func in CORNER_GENERATORS:
    for file in glob.glob(INPUT_FILES):
        image = cv.imread(file, cv.IMREAD_COLOR)
        greyscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        corners = func(greyscale)
        result = image
        for corner in corners.astype(numpy.int):
            x, y = corner.ravel()
            result = cv.drawMarker(result, (x, y), (0, 0, 255))
        filename = os.path.splitext(os.path.basename(file))[0]
        output_file = OUTPUT_FILE.format(filename, alg_name)
        cv.imwrite(output_file, result)
