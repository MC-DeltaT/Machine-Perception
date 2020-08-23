import glob
from math import pi, tau
import os.path

import cv2 as cv
import numpy


INPUT_FILES = 'data/prac03ex03img*'
OUTPUT_FILE = 'output/{}-accumulator.png'

CANNY_THRESHOLD1 = 100
CANNY_THRESHOLD2 = 200

R_STEPS = 1000
THETA_STEPS = 1000

for file in glob.glob(INPUT_FILES):
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    g_x = cv.Sobel(image, cv.CV_16SC1, 1, 0)
    g_y = cv.Sobel(image, cv.CV_16SC1, 0, 1)
    edge = cv.Canny(g_x, g_y, CANNY_THRESHOLD1, CANNY_THRESHOLD2, L2gradient=True)
    edge_mask = edge < 127
    g_x = numpy.ma.masked_array(g_x, edge_mask)
    g_y = numpy.ma.masked_array(g_y, edge_mask)
    theta = numpy.arctan2(g_x, g_y)
    n = numpy.stack([g_x, g_y], axis=2).astype(numpy.float)
    n /= numpy.linalg.norm(n, axis=2)[:, :, numpy.newaxis]
    y, x = numpy.indices(image.shape)
    r = x * n[:, :, 0] + y * n[:, :, 1]
    r_min = numpy.min(r)
    r_max = numpy.max(r)
    accumulator = numpy.zeros((R_STEPS, THETA_STEPS), numpy.uint8)
    for i, j in numpy.ndindex(*image.shape):
        if not edge_mask[i, j]:
            acc_i = int((R_STEPS - 1) * (r[i, j] - r_min) / (r_max - r_min))
            acc_j = int((THETA_STEPS - 1) * (theta[i, j] + pi) / tau)
            accumulator[acc_i, acc_j] += 1
    accumulator = cv.equalizeHist(accumulator)
    filename = os.path.splitext(os.path.basename(file))[0]
    output_file = OUTPUT_FILE.format(filename)
    cv.imwrite(output_file, accumulator)
