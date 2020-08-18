import glob
import os.path

import cv2 as cv
import numpy


INPUT_FILES = 'data/prac02ex02img*'
OUTPUT_FILE = 'output/{}-{}.png'

KERNELS = [
    ('prewitt_x', numpy.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])),
    ('prewitt_y', numpy.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])),
    ('sobel_x', numpy.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])),
    ('sobel_y', numpy.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])),
    ('laplacian', numpy.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])),
    ('gaussian', numpy.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]]) / 273.0)
]

for file in glob.glob(INPUT_FILES):
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    for kernel_name, kernel in KERNELS:
        kernel = cv.flip(kernel, -1)
        result = cv.filter2D(image, cv.CV_32F, kernel)
        result = cv.normalize(result, None, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        filename = os.path.splitext(os.path.basename(file))[0]
        output_file = OUTPUT_FILE.format(filename, kernel_name)
        cv.imwrite(output_file, result)
