# Defines the descriptor used for digit recognition.

import cv2 as cv


__all__ = [
    'digit_descriptor'
]


# Produces a feature descriptor for an digit image that is used for recognition.
def digit_descriptor(image):
    # Convert to greyscale, colour information isn't that important (always light digit on dark
    # background).
    if image.shape == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Crop the image to just the foreground area.
    _, binary = cv.threshold(image, 0, 255, cv.THRESH_OTSU)
    x, y, w, h = cv.boundingRect(binary)
    image = image[y:y + h, x:x + w]

    # Resize to uniform size.
    image = cv.resize(image, (32, 32), interpolation=cv.INTER_CUBIC)
    image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
    # Threshold to remove background.
    _, image = cv.threshold(image, 0, 255, cv.THRESH_OTSU + cv.THRESH_TOZERO)
    # Use HOG descriptor, which is better at handling variations in the image than raw image data.
    descriptor = _hog.compute(image).ravel()
    return descriptor


_hog = cv.HOGDescriptor((16, 16), (16, 16), (8, 8), (8, 8), 9)
