# Functions for extracting the digits and house numbers.

import cv2 as cv
from itertools import product
from math import atan2, hypot, radians
import numpy
from typing import Sequence, Tuple


__all__ = [
    'detect_regions',
    'filter_regions',
    'select_number'
]


BoundingBox = Tuple[float, float, float, float]
Point = Tuple[float, float]
Line = Tuple[Point, Point]


# Finds MSERS.
def detect_regions(image_grey: numpy.ndarray):
    max_area = int(image_grey.shape[0] * image_grey.shape[1] * REGION_AREA_MAX)
    mser = cv.MSER_create(MSER_DELTA, REGION_AREA_MIN, max_area)
    msers, boxes = mser.detectRegions(image_grey)
    return boxes, msers


# Finds regions that are possibly digits.
def filter_regions(image_grey: numpy.ndarray, region_boxes: Sequence[BoundingBox],
                   region_points: numpy.ndarray) -> Sequence[BoundingBox]:
    max_area = int(image_grey.shape[0] * image_grey.shape[1] * REGION_AREA_MAX)

    adjusted_boxes = []
    for box, points in zip(region_boxes, region_points):
        x, y, w, h = box

        # OpenCV's MSER area bounds don't seem to always work.
        if not REGION_AREA_MIN <= w * h <= max_area:
            continue

        # Filter out regions with aspect ratios unlikely to occur for digits.
        if not REGION_ASPECT_RATIO_RANGE[0] <= w / h <= REGION_ASPECT_RATIO_RANGE[1]:
            continue

        region_grey = image_grey[y:y + h, x:x + w]

        # Further processing and recognition assumes that the digit is the foreground (i.e. light on
        # a dark background) in the local area. But MSER can produce regions that are dark on a
        # light background, so need to filter out those.
        if not _region_is_foreground(region_grey, box, points):
            continue

        # Digits typically only have 1 main foreground component.
        if _has_multiple_components(region_grey):
            continue

        # Filters out regions whose foreground is not isolated.
        is_isolated, box = _is_foreground_isolated(image_grey, box)
        if not is_isolated:
            continue

        adjusted_boxes.append(box)

    # Remove boxes are are basically the same.
    adjusted_boxes = _remove_equivalent_boxes(adjusted_boxes)

    return adjusted_boxes


# Selects the regions (from detect_regions()) that form the house number.
def select_number(boxes: Sequence[BoundingBox]) -> Sequence[BoundingBox]:
    if len(boxes) <= 1:
        return boxes

    boxes = sorted(boxes, key=lambda b: b[0])

    paths = []
    for i, box1 in enumerate(boxes):
        p = [box1]
        c1x = box1[0] + box1[2] / 2
        c1y = box1[1] + box1[3] / 2
        for j in range(i + 1, len(boxes)):
            box2 = boxes[j]
            c2x = box2[0] + box2[2] / 2
            c2y = box2[1] + box2[3] / 2
            # Create only roughly horizontal paths.
            if abs(atan2(c2y - c1y, c2x - c1x)) <= NUMBER_PATH_ANGLE_THRESHOLD:
                # Select only boxes that are near to the previous box in the X axis.
                if c2x - (p[-1][0] + p[-1][2] / 2) <= 3 * max(b[2] for b in p):
                    # Select only boxes that have similar height to the previous box.
                    if abs(p[-1][3] - box2[3]) / box2[3] <= NUMBER_PATH_HEIGHT_DIFF_THRESHOLD:
                        p.append(box2)
                    else:
                        break
                else:
                    break
        paths.append(p)
    path_lengths = [len(p) for p in paths]
    box_heights = [numpy.array([b[3] for b in p]) for p in paths]
    box_height_means = numpy.array([numpy.mean(hs) for hs in box_heights])

    path_filter = numpy.zeros(len(paths), numpy.bool)
    # Consider only the paths with the tallest boxes.
    path_filter[box_height_means / numpy.max(box_height_means) < NUMBER_TALL_THRESHOLD] = True
    # Then select the longest path.
    best_path = numpy.ma.argmax(numpy.ma.masked_array(path_lengths, path_filter))
    return paths[best_path]


# Checks if a significant amount of pixels in a region are foreground when considering the local area.
def _region_is_foreground(region_grey: numpy.ndarray, box: BoundingBox, points: numpy.ndarray) -> bool:
    _, binary = cv.threshold(region_grey, 0, 1, cv.THRESH_OTSU)
    indices = numpy.flip(points - [box[0], box[1]], 1)
    fg_points = binary.ravel()[numpy.ravel_multi_index((indices[:, 0], indices[:, 1]), binary.shape)]
    region_fg_ratio = numpy.average(fg_points)
    return region_fg_ratio >= REGION_FG_RATIO_THRESHOLD


# Checks if there are multiple (significant) foreground components.
def _has_multiple_components(region_grey: numpy.ndarray) -> bool:
    _, binary = cv.threshold(region_grey, 0, 255, cv.THRESH_OTSU)
    _, labels = cv.connectedComponents(binary)
    fg_component_areas = numpy.bincount(labels.ravel())[1:]
    max_area = numpy.max(fg_component_areas)
    significant_components = fg_component_areas / max_area > REGION_SIGNIFICANT_COMPONENT_AREA_THRESHOLD
    return numpy.count_nonzero(significant_components) > 1


# Checks if a region's foreground is not part of a larger object.
def _is_foreground_isolated(image_grey: numpy.ndarray, box: BoundingBox) -> Tuple[bool, BoundingBox]:
    x1, y1, w, h = box
    x2 = x1 + w
    y2 = y1 + h
    is_isolated = True
    # Try to grow region side by side until it covers entire foreground area.
    for axis, direction in product(((1, 0), (0, 1)), (-1, 1)):
        for _ in range(0, REGION_GROWTH + 1):
            prev_x1 = x1
            prev_y1 = y1
            prev_x2 = x2
            prev_y2 = y2
            if direction < 0:
                x1 = max(x1 - axis[0], 0)
                y1 = max(y1 - axis[1], 0)
            elif direction > 0:
                x2 = min(x2 + axis[0], image_grey.shape[1])
                y2 = min(y2 + axis[1], image_grey.shape[0])

            region_grey = image_grey[y1:y2, x1:x2]

            _, binary = cv.threshold(region_grey, 0, 1, cv.THRESH_OTSU)
            # Find the label of the main foreground component.
            _, labels = cv.connectedComponents(binary)
            fg_component_areas = numpy.bincount(labels.ravel())[1:]
            fg_label = numpy.argmax(fg_component_areas) + 1

            # Check if the main foreground component touches the border of the box.
            border_mask = numpy.zeros_like(region_grey, numpy.bool)
            if direction < 0 and axis[0]:
                border_mask[:, 0] = True
            elif direction > 0 and axis[0]:
                border_mask[:, -1] = True
            elif direction < 0 and axis[1]:
                border_mask[0, :] = True
            elif direction > 0 and axis[1]:
                border_mask[-1, :] = True
            border_labels = labels[border_mask]
            if numpy.all(border_labels != fg_label):
                # Isolated in this axis and direction.
                # Revert box so we don't get a 1 pixel gap around the foreground.
                x1 = prev_x1
                y1 = prev_y1
                x2 = prev_x2
                y2 = prev_y2
                break
        else:
            is_isolated = False
            break
    return is_isolated, (x1, y1, x2 - x1, y2 - y1)


# Find all the groups of boxes that are effectively the same, and from those groups chooses only
# the regions with minimal area.
def _remove_equivalent_boxes(boxes: Sequence[BoundingBox]) -> Sequence[BoundingBox]:
    boxes = list(set(boxes))
    centres = [(x + w / 2, y + h / 2) for x, y, w, h in boxes]
    equivalencies = []
    for i, (box1, (c1x, c1y)) in enumerate(zip(boxes, centres)):
        tmp = {box1}
        for j, (box2, (c2x, c2y)) in enumerate(zip(boxes, centres)):
            if box1 is not box2:
                # Consider boxes to be the same if their centres are close.
                distance = hypot(c1x - c2x, c1y - c2y)
                if distance < EQUIVALENT_BOX_DISTANCE:
                    tmp.add(box2)
        equivalencies.append(tmp)
    for _ in equivalencies:
        for e1 in equivalencies:
            for e2 in equivalencies:
                if e1 != e2:
                    if any(b in e1 for b in e2):
                        for b in e2:
                            e1.add(b)
    boxes = list(set(min(e, key=lambda b: b[2] * b[3]) for e in equivalencies))
    return boxes


MSER_DELTA = 15
REGION_AREA_MIN = 50        # Pixels
REGION_AREA_MAX = 0.5 * 0.5     # Fraction of image area
REGION_ASPECT_RATIO_RANGE = (0.2, 0.8)
REGION_FG_RATIO_THRESHOLD = 0.75
REGION_SIGNIFICANT_COMPONENT_AREA_THRESHOLD = 0.05   # Fraction of largest component's area
REGION_GROWTH = 5       # Pixels
EQUIVALENT_BOX_DISTANCE = 5     # Pixels
NUMBER_PATH_ANGLE_THRESHOLD = radians(20)
NUMBER_PATH_HEIGHT_DIFF_THRESHOLD = 0.25    # Fraction of current box
NUMBER_TALL_THRESHOLD = 0.75    # Fraction of tallest boxes' height
