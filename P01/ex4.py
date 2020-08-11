from math import atan, degrees

import cv2
import numpy


INPUT_FILE = 'data/prac01ex04img02.png'
OUTPUT_FILE = 'output/prac01ex04img02-rotated.png'

image = cv2.imread(INPUT_FILE)
height, width, _ = image.shape

s1 = next(i for i, pixel in enumerate(image[0]) if any(pixel > 0))
s2 = next(i for i, row in enumerate(image) if any(row[0] > 0))
angle = degrees(atan(s1 / s2))

old_tl = [s1, 0, 1]
old_br = [width - s1, height, 1]

centre = ((width - 1) / 2, (height - 1) / 2)
rotation_matrix = cv2.getRotationMatrix2D(centre, angle, 1)

new_tl = numpy.dot(rotation_matrix, old_tl).round().astype(numpy.int)
new_br = numpy.dot(rotation_matrix, old_br).round().astype(numpy.int)

image = cv2.warpAffine(image, rotation_matrix, (width, height))
image = image[new_tl[1]:new_br[1], new_tl[0]:new_br[0]]
cv2.imwrite(OUTPUT_FILE, image)
