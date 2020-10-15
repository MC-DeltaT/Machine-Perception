# Functions for extracting the digits and house numbers.

import cv2 as cv
from itertools import product
from math import hypot, radians
import numpy


__all__ = [
    'detect_regions',
    'select_number'
]


# Finds regions that are possibly digits.
def detect_regions(image):
    image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    msers, boxes = _mser.detectRegions(image_grey)

    adjusted_boxes = []
    for box, points in zip(boxes, msers):
        x, y, w, h = box

        # Filter out tiny and huge regions, unlikely to be digits.
        if not 100 < w * h < image.shape[0] * image.shape[1] // 4:
            continue

        # Filter out regions with aspect ratios unlikely to occur for digits.
        if not REGION_ASPECT_RATIO_RANGE[0] <= w / h <= REGION_ASPECT_RATIO_RANGE[1]:
            continue

        # Further processing and recognition assumes that the digit is the foreground (i.e. light on
        # a dark background) in the local area. But MSER can produce regions that are dark on a
        # light background, so need to filter out those.
        if not _region_is_foreground(image_grey, box, points):
            continue

        # Filters out regions that are sub-regions of other regions (occurs very often with MSER).
        is_subregion, box = _is_subregion(image_grey, box)
        if is_subregion:
            continue

        adjusted_boxes.append(box)

    # Remove boxes are are basically the same.
    adjusted_boxes = _remove_equivalent_boxes(adjusted_boxes)

    return adjusted_boxes


# Selects the regions (from detect_regions()) that form the house number.
def select_number(image, boxes):
    if len(boxes) <= 1:
        return boxes

    # TODO? drop lines with any boxes that differ in colour significantly

    # Find all possible line segments with endpoints at box centres.
    lines = []
    for i, box1 in enumerate(boxes):
        c1x = box1[0] + box1[2] / 2
        c1y = box1[1] + box1[3] / 2
        for j, box2 in enumerate(boxes):
            if i != j:
                c2x = box2[0] + box2[2] / 2
                c2y = box2[1] + box2[3] / 2
                lines.append(((c1x, c1y), (c2x, c2y)))
    lines = numpy.array(lines)

    line_angles = numpy.arctan2(lines[:, 1, 1] - lines[:, 0, 1], lines[:, 1, 0] - lines[:, 0, 0])

    # Associate each line with a set of boxes that intersect it.
    intersecting_boxes = [_boxes_intersecting_line(l, boxes) for l in lines]
    # Sort boxes by x coordinate, since digits run from left to right.
    intersecting_boxes = [sorted(bs, key=lambda b: b[0]) for bs in intersecting_boxes]

    box_counts = [len(bs) for bs in intersecting_boxes]

    box_xs = [numpy.array([b[0] + b[2] / 2 for b in bs]) for bs in intersecting_boxes]
    x_diffs = [numpy.diff(xs) for xs in box_xs]
    x_diff_diffs = [numpy.diff(xds) for xds in x_diffs]

    # Compute various statistics on the lines' box heights.
    heights = [numpy.array([b[3] for b in bs]) for bs in intersecting_boxes]
    height_means = numpy.array([numpy.mean(hs) for hs in heights])
    height_diffs = [numpy.diff(hs) for hs in heights]

    line_filter = numpy.zeros(len(lines), numpy.bool)
    # Select only roughly horizontal lines.
    line_filter[numpy.abs(line_angles) > NUMBER_LINE_ANGLE_THRESHOLD] = True
    # Remove lines whose boxes have large height variation.
    has_height_outliers = numpy.array(
        [numpy.any(numpy.abs(hds) / hs[:-1] > NUMBER_BOX_HEIGHT_DIFF_THRESHOLD)
         for hs, hds in zip(heights, height_diffs)])
    line_filter[has_height_outliers] = True
    # Remove lines whose boxes are not spaced evenly in the X axis.
    has_x_outliers = numpy.array(
        [numpy.any(numpy.abs(xdds) / xds[:-1] > NUMBER_BOX_X_DIFF_THRESHOLD)
         for xds, xdds in zip(x_diffs, x_diff_diffs)])
    line_filter[has_x_outliers] = True
    # Select the lines with the tallest boxes (assume the house number digits are always bigger than
    # any other text).
    line_filter[height_means / numpy.ma.max(numpy.ma.masked_array(height_means, line_filter)) < NUMBER_TALL_THRESHOLD] = True
    # Select the line with the most boxes.
    line_filter[box_counts < numpy.ma.max(numpy.ma.masked_array(box_counts, line_filter))] = True
    if numpy.all(line_filter):
        return None
    else:
        best_line = numpy.argmin(line_filter)
        return intersecting_boxes[best_line]


REGION_ASPECT_RATIO_RANGE = (0.2, 0.8)
REGION_FG_RATIO_THRESHOLD = 0.5
REGION_GROWTH = 5
EQUIVALENT_BOX_DISTANCE = 5
NUMBER_BOX_X_DIFF_THRESHOLD = 0.5
NUMBER_BOX_HEIGHT_DIFF_THRESHOLD = 0.25
NUMBER_LINE_ANGLE_THRESHOLD = radians(20)
NUMBER_TALL_THRESHOLD = 0.8


# Checks if enough pixels in a region are foreground when considering the local area.
def _region_is_foreground(image_grey, box, points):
    x, y, w, h = box
    region_grey = image_grey[y:y + h, x:x + w]
    _, binary = cv.threshold(region_grey, 0, 1, cv.THRESH_OTSU)
    indices = numpy.flip(points - [x, y], 1)
    fg_points = binary.ravel()[numpy.ravel_multi_index((indices[:, 0], indices[:, 1]), binary.shape)]
    region_fg_ratio = numpy.average(fg_points)
    return region_fg_ratio >= REGION_FG_RATIO_THRESHOLD


# Checks if a region's foreground is part of a larger object.
def _is_subregion(image_grey, box):
    x1, y1, w, h = box
    x2 = x1 + w
    y2 = y1 + h
    is_subregion = False
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
            region_grey = cv.normalize(region_grey, None, 0, 255, cv.NORM_MINMAX)

            _, binary = cv.threshold(region_grey, 0, 1, cv.THRESH_OTSU)
            # binary = cv.medianBlur(binary, 3)
            # Find label of foreground point closest the to centre of the region.
            fg_indices = numpy.argwhere(binary)
            centre_index = (binary.shape[0] / 2, binary.shape[1] / 2)
            fg_centre_index = fg_indices[
                numpy.argmin(numpy.linalg.norm(fg_indices - centre_index, axis=1))]
            _, labels = cv.connectedComponents(binary)
            centre_label = labels[fg_centre_index[0], fg_centre_index[1]]
            assert centre_label > 0

            # Check if the centre foreground component touches the border of the box.
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
            if numpy.all(border_labels != centre_label):
                # Not subregion is this axis and direction.
                # Revert box so we don't get a 1 pixel gap around the foreground.
                x1 = prev_x1
                y1 = prev_y1
                x2 = prev_x2
                y2 = prev_y2
                break
        else:
            is_subregion = True
            break
    return is_subregion, (x1, y1, x2 - x1, y2 - y1)


# Find all the groups of boxes that are effectively the same, and from those groups chooses only
# the regions with minimal area.
def _remove_equivalent_boxes(boxes):
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
    boxes = set(min(e, key=lambda b: b[2] * b[3]) for e in equivalencies)
    return boxes


# Finds all boxes that intersect or contain a line segment.
def _boxes_intersecting_line(line, boxes):
    # Checks if a point lies within a box.
    def point_in_box(point, box):
        return box[0] <= point[0] <= box[0] + box[2] and box[1] <= point[1] <= box[1] + box[3]

    # Checks if 2 line segments (defined by endpoints) intersect.
    def lines_intersect(line1, line2):
        # Solve for line parameters s, t for intersection.
        d1 = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1])
        d2 = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1])
        a = [[d1[0], -d2[0]],
             [d1[1], -d2[1]]]
        b = [line2[0][0] - line1[0][0], line2[0][1] - line1[0][1]]
        try:
            s, t = numpy.linalg.solve(a, b)
        except numpy.linalg.LinAlgError:
            # No solution, no intersection.
            return False
        else:
            # Intersection if solution lies within the line segments
            return 0 <= s <= 1 and 0 <= t <= 1

    intersecting_boxes = set()
    for box in boxes:
        if point_in_box(line[0], box) or point_in_box(line[1], box):
            intersecting_boxes.add(box)
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h
        box_lines = [((x1, y1), (x2, y1)),
                     ((x1, y1), (x1, y2)),
                     ((x1, y2), (x2, y2)),
                     ((x2, y1), (x2, y2))]
        if any(lines_intersect(l, line) for l in box_lines):
            intersecting_boxes.add(box)
    if len(intersecting_boxes) == 0: breakpoint()
    return list(intersecting_boxes)


_mser = cv.MSER_create()
