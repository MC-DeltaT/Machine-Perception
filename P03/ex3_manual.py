import glob
from math import cos, pi, sin, tau
import os.path

import cv2 as cv
import numpy


INPUT_FILES = 'data/prac03ex03img*'
ACC_OUTPUT_FILE = 'output/{}-accumulator.png'
LINE_OUTPUT_FILE = 'output/{}-lines.png'

CANNY_THRESHOLD1 = 100
CANNY_THRESHOLD2 = 200

R_STEPS = 100
THETA_STEPS = 500
ACC_THRESHOLD = 15

for file in glob.glob(INPUT_FILES):
    image = cv.imread(file, cv.IMREAD_COLOR)
    greyscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    g_x = cv.Sobel(greyscale, cv.CV_16SC1, 1, 0)
    g_y = cv.Sobel(greyscale, cv.CV_16SC1, 0, 1)
    edge = cv.Canny(g_x, g_y, CANNY_THRESHOLD1, CANNY_THRESHOLD2, L2gradient=True)

    g_x = g_x.astype(numpy.float)
    g_y = g_y.astype(numpy.float)
    edge_mask = edge < 127
    g_x = numpy.ma.masked_array(g_x, edge_mask)
    g_y = numpy.ma.masked_array(g_y, edge_mask)

    theta = numpy.ma.arctan2(g_y, g_x)
    n = numpy.ma.stack([g_x, g_y], axis=2)
    n /= numpy.linalg.norm(n, axis=2)[:, :, numpy.newaxis]
    y, x = numpy.indices(greyscale.shape)
    r = x * n[:, :, 0] + y * n[:, :, 1]

    r_min = numpy.min(r)
    r_max = numpy.max(r)
    accumulator = numpy.zeros((R_STEPS, THETA_STEPS), numpy.uint8)
    acc_i = ((R_STEPS - 1) * (r - r_min) / (r_max - r_min)).astype(numpy.int)
    acc_j = ((THETA_STEPS - 1) * (theta + pi) / tau).astype(numpy.int)
    for i, j in numpy.ndindex(*greyscale.shape):
        if not edge_mask[i, j]:
            accumulator[acc_i[i, j], acc_j[i, j]] += 1

    filename = os.path.splitext(os.path.basename(file))[0]
    output = cv.equalizeHist(accumulator)
    output_file = ACC_OUTPUT_FILE.format(filename)
    cv.imwrite(output_file, output)

    output = image
    is_line = accumulator >= ACC_THRESHOLD
    lines = []
    for i, j in numpy.ndindex(*accumulator.shape):
        if is_line[i, j]:
            r_ = (r_max - r_min) * i / (R_STEPS - 1) + r_min
            theta_ = tau * j / (THETA_STEPS - 1) - pi
            lines.append((r_, theta_))

    for r, theta in lines:
        a = cos(theta)
        b = sin(theta)
        x0 = a * r
        y0 = b * r
        pt1 = (int(x0 - 1000 * b), int(y0 + 1000 * a))
        pt2 = (int(x0 + 1000 * b), int(y0 - 1000 * a))
        output = cv.line(output, pt1, pt2, (0, 0, 255), 2, cv.LINE_AA)
    output_file = LINE_OUTPUT_FILE.format(filename)
    cv.imwrite(output_file, output)
