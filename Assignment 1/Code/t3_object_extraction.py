import cv2 as cv
import numpy
import os.path
from pathlib import Path
from random import randrange

def threshold_card(image):
    # Assume the background includes the black behind/around the card and the card's white bit.
    image = image.astype(numpy.uint16)
    # Foreground is pixels with red > blue and red > green.
    binary = numpy.logical_and(image[:, :, 2] > image[:, :, 0] * 2, image[:, :, 2] > image[:, :, 1] * 2)
    return binary.astype(numpy.uint8)

def threshold_dugong(image):
    # Median blur to get rid of small speckled bits in ocean.
    image = cv.medianBlur(image, 3)
    image = image.astype(numpy.uint16)
    # Background is pixels with blue > red and green > red.
    binary = numpy.logical_and(image[:, :, 0] > image[:, :, 2] * 1.8, image[:, :, 1] > image[:, :, 2] * 2.2)
    binary = numpy.logical_not(binary)
    return binary.astype(numpy.uint8)

INPUTS = [
    ('data/card.png', threshold_card),
    ('data/dugong.jpg', threshold_dugong)
]
OUTPUT_DIR = 'results/{}'
BINARY_OUTPUT_FILE = 'binary.png'
OBJECTS_OUTPUT_FILE = 'objects.png'
AREAS_OUTPUT_FILE = 'object_areas.txt'

for file, threshold_func in INPUTS:
    image = cv.imread(file, cv.IMREAD_COLOR)
    filename = os.path.splitext(os.path.basename(file))[0]
    output_dir = OUTPUT_DIR.format(filename)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    binary = threshold_func(image)

    output_file = os.path.join(output_dir, BINARY_OUTPUT_FILE)
    cv.imwrite(output_file, binary * 255)

    label_count, labels = cv.connectedComponents(binary, connectivity=8)

    colours = [(255, 255, 255)]
    colours.extend((randrange(0, 200), randrange(0, 200), randrange(0, 200)) for _ in range(1, label_count))
    result = numpy.array(colours)[labels]
    output_file = os.path.join(output_dir, OBJECTS_OUTPUT_FILE)
    cv.imwrite(output_file, result)

    # TODO: some way to associate areas with objects
    objects = [list(zip(*numpy.where(labels == l))) for l in range(1, label_count)]
    with open(os.path.join(output_dir, AREAS_OUTPUT_FILE), 'w') as areas_file:
        for i, object in enumerate(objects, 1):
            area = len(object)
            areas_file.write(f'{i}: {area}\n')
