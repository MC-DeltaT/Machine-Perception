import glob
import os.path

import cv2 as cv
import numpy


INPUT_FILES = 'data/prac03ex01img*'
OUTPUT_FILE = 'output/{}-{}_response.png'

WINDOW_SIZE = (5, 5)
WINDOW_SIGMA = 1
HARRIS_K = 0.05


def harris(A):
    return numpy.linalg.det(A) - HARRIS_K * numpy.trace(A, axis1=2, axis2=3) ** 2


def shi_tomasi(A):
    eigenvalues = numpy.linalg.eigvals(A)
    return numpy.min(eigenvalues, axis=2)


ALGORITHMS = [
    ('harris', harris),
    ('shi_tomasi', shi_tomasi)
]


for file in glob.glob(INPUT_FILES):
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    for algorithm_name, algorithm in ALGORITHMS:
        gradient_x = cv.Sobel(image, cv.CV_32F, 1, 0)
        gradient_y = cv.Sobel(image, cv.CV_32F, 0, 1)
        I_x2 = gradient_x ** 2
        I_y2 = gradient_y ** 2
        I_xy = gradient_x * gradient_y
        I_x2_w = cv.GaussianBlur(I_x2, WINDOW_SIZE, WINDOW_SIGMA)
        I_y2_w = cv.GaussianBlur(I_y2, WINDOW_SIZE, WINDOW_SIGMA)
        I_xy_w = cv.GaussianBlur(I_xy, WINDOW_SIZE, WINDOW_SIGMA)
        A = numpy.empty(image.shape + (2, 2))
        for i, j in numpy.ndindex(*image.shape):
            A[i, j] = numpy.array([
                [I_x2_w[i, j], I_xy_w[i, j]],
                [I_xy_w[i, j], I_y2_w[i, j]]
            ])
        result = algorithm(A)
        result = cv.normalize(result, None, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        filename = os.path.splitext(os.path.basename(file))[0]
        output_file = OUTPUT_FILE.format(filename, algorithm_name)
        cv.imwrite(output_file, result)
