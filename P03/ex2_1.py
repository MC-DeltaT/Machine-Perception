import glob
import os.path

import cv2 as cv
import numpy


INPUT_FILES = 'data/prac03ex02img*'
OUTPUT_FILE = 'output/{}-gradient.png'

for file in glob.glob(INPUT_FILES):
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    gradient_x = cv.Sobel(image, cv.CV_32F, 1, 0)
    gradient_y = cv.Sobel(image, cv.CV_32F, 0, 1)
    result = numpy.sqrt(gradient_x ** 2 + gradient_y ** 2)
    result = cv.normalize(result, None, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    filename = os.path.splitext(os.path.basename(file))[0]
    output_file = OUTPUT_FILE.format(filename)
    cv.imwrite(output_file, result)
