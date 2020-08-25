import glob
import os.path

import cv2 as cv


INPUT_FILES = 'data/prac03ex04img*'
OUTPUT_FILE = 'output/{}-regions.png'


for file in glob.glob(INPUT_FILES):
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    mser = cv.MSER_create()
    regions, boxes = mser.detectRegions(image)

    hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    result = cv.polylines(image, hulls, 1, (0, 255, 0))

    filename = os.path.splitext(os.path.basename(file))[0]
    output_file = OUTPUT_FILE.format(filename)
    cv.imwrite(output_file, result)
