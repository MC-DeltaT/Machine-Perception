# Defines the descriptor used for digit recognition.

import cv2 as cv
import numpy


__all__ = [
    'digit_descriptor'
]


# Produces a feature descriptor for an digit image that is used for recognition.
def digit_descriptor(image_grey: numpy.ndarray) -> numpy.ndarray:
    # Crop the image to just the foreground area.
    _, binary = cv.threshold(image_grey, 0, 255, cv.THRESH_OTSU)
    x, y, w, h = cv.boundingRect(binary)
    image_grey = image_grey[y:y + h, x:x + w]

    # Resize to uniform size.
    image_grey = cv.resize(image_grey, (32, 32), interpolation=cv.INTER_CUBIC)
    image_grey = cv.normalize(image_grey, None, 0, 255, cv.NORM_MINMAX)
    # Threshold to remove background. Retain foreground for better representation of the digit.
    _, image_grey = cv.threshold(image_grey, 0, 255, cv.THRESH_OTSU + cv.THRESH_TOZERO)
    # Use HOG descriptor, which is better at handling variations in the image than raw image data.
    descriptor = _hog.compute(image_grey).ravel()
    return descriptor


_hog = cv.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)
