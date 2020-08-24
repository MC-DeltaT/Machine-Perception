import glob
import os.path

import cv2 as cv


INPUT_FILES = 'data/prac03ex04img*'
OUTPUT_FILE = 'output/{}-regions.png'


for file in glob.glob(INPUT_FILES):
    image = cv.imread(file, cv.IMREAD_COLOR)
    mser = cv.MSER_create(_min_area=10, _delta=3)
    regions, boxes = mser.detectRegions(image)

    result = image
    for region in regions:
        cv.fillPoly(result, [region], (0, 255, 0))

    filename = os.path.splitext(os.path.basename(file))[0]
    output_file = OUTPUT_FILE.format(filename)
    cv.imwrite(output_file, result)
