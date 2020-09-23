import cv2 as cv
import numpy
import os.path
from pathlib import Path

def threshold_card(image):
    # Assume the background includes the black behind/around the card and the card's white bit.
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    binary = image[:, :, 1] > 100
    return binary.astype(numpy.uint8)

def threshold_dugong(image):
    # Median blur to get rid of small speckled bits in ocean.
    image = cv.medianBlur(image, 3)
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    binary = image[:, :, 1] < 150
    return binary.astype(numpy.uint8)

INPUTS = [
    ('data/card.png', threshold_card),
    ('data/dugong.jpg', threshold_dugong)
]
OUTPUT_DIR = 'results/{}'
BINARY_OUTPUT_FILE = 'binary.png'
LABELS_OUTPUT_FILE = 'object_labels.png'
OBJECTS_OUTPUT_FILE = 'objects.png'
AREAS_OUTPUT_FILE = 'object_areas.txt'
LABEL_COLOURS = numpy.array([
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 150, 255),
    (255, 0, 255)
])
PADDING = 2

for file, threshold_func in INPUTS:
    image = cv.imread(file, cv.IMREAD_COLOR)
    filename = os.path.splitext(os.path.basename(file))[0]
    output_dir = OUTPUT_DIR.format(filename)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    binary = threshold_func(image)

    output_file = os.path.join(output_dir, BINARY_OUTPUT_FILE)
    cv.imwrite(output_file, binary * 255)

    label_count, labels = cv.connectedComponents(binary, connectivity=8)

    result = LABEL_COLOURS[labels]
    output_file = os.path.join(output_dir, LABELS_OUTPUT_FILE)
    cv.imwrite(output_file, result)

    # Group pixels by label.
    points_by_label = [numpy.array(list(zip(*numpy.where(labels == l)))) for l in range(1, label_count)]
    # Sort objects by area.
    points_by_label = sorted(points_by_label, key=lambda ps: ps.shape[0], reverse=True)

    # Extract the objects and collate them.
    bounding_boxes = [cv.boundingRect(ps[:, [1, 0]]) for ps in points_by_label]
    largest_width = max(width for x, y, width, height in bounding_boxes)
    total_height = sum(height for x, y, width, height in bounding_boxes)
    result = numpy.full((total_height + PADDING * label_count - 1, largest_width, 3), 255, numpy.uint8)
    y = 0
    for (box_x, box_y, box_width, box_height) in bounding_boxes:
        src_index = (slice(box_y, box_y + box_height), slice(box_x, box_x + box_width))
        obj = image[src_index] * numpy.repeat(binary[src_index][:, :, numpy.newaxis], 3, axis=2)
        margin = (largest_width - box_width) // 2
        result[y:y + box_height, margin:box_width + margin] = obj
        y += box_height + PADDING
    output_file = os.path.join(output_dir, OBJECTS_OUTPUT_FILE)
    cv.imwrite(output_file, result)

    with open(os.path.join(output_dir, AREAS_OUTPUT_FILE), 'w') as areas_file:
        areas_file.writelines(f'{ps.shape[0]}\n' for ps in points_by_label)
