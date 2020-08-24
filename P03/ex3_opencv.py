import glob
from math import cos, pi, sin
import os.path

import cv2 as cv


INPUT_FILES = 'data/prac03ex03img*'
OUTPUT_FILE = 'output/{}-lines.png'

CANNY_THRESHOLD1 = 200
CANNY_THRESHOLD2 = 300

R_RESOLUTION = 1
THETA_RESOLUTION = pi / 100
ACC_THRESHOLD = 150

for file in glob.glob(INPUT_FILES):
    image = cv.imread(file, cv.IMREAD_COLOR)
    greyscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edge = cv.Canny(greyscale, CANNY_THRESHOLD1, CANNY_THRESHOLD2, L2gradient=True)
    lines = cv.HoughLines(edge, R_RESOLUTION, THETA_RESOLUTION, ACC_THRESHOLD)

    output = image
    if lines is not None:
        lines = lines[:, 0, :]
        for r, theta in lines:
            a = cos(theta)
            b = sin(theta)
            x0 = a * r
            y0 = b * r
            pt1 = (int(x0 - 1000 * b), int(y0 + 1000 * a))
            pt2 = (int(x0 + 1000 * b), int(y0 - 1000 * a))
            output = cv.line(output, pt1, pt2, (0, 0, 255), 2, cv.LINE_AA)

    filename = os.path.splitext(os.path.basename(file))[0]
    output_file = OUTPUT_FILE.format(filename)
    cv.imwrite(output_file, output)
