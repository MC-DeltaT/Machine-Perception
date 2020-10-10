import cv2 as cv
from glob import glob
from math import cos, sin, pi
import numpy
import os.path

for input_file in glob('train/*.*'):
    image = cv.imread(input_file, cv.IMREAD_COLOR)
    image = cv.medianBlur(image, 3)
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    grey = cv.normalize(grey, None, 0, 255, cv.NORM_MINMAX)

    # TODO: proper markers with connected components or whatever
    background = grey < 50
    foreground = grey > 200
    markers = numpy.full(image.shape[:-1], 127, dtype=numpy.int32)
    markers[background] = 0
    markers[foreground] = 255

    cv.watershed(image, markers)

    filename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f'output/{filename}.png'
    cv.imwrite(output_file, markers)
