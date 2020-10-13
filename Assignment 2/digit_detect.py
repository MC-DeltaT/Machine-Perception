import cv2 as cv
from math import hypot, radians
import numpy


__all__ = [
    'detect_regions',
    'select_number'
]


# Finds regions that are possibly digits.
def detect_regions(image):
    image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    mser = cv.MSER_create()
    _, boxes = mser.detectRegions(image_grey)

    adjusted_boxes = []
    for x, y, w, h in boxes:
        # Filter out tiny and huge regions, unlikely to be digits.
        if not 100 < w * h < image.shape[0] * image.shape[1] // 4:
            continue

        # Digits typically have aspect ratios < 1, but not too close to 0.
        if not 0.2 < w / h < 0.8:
            continue

        # Try to grow region until it covers entire foreground area. This filters out regions that
        # are sub-regions of other regions (occurs very often with MSER).
        # TODO: grow until centre connected component is covered
        ok = False
        for growth in range(0, 6):
            x1 = max(0, x - growth)
            y1 = max(0, y - growth)
            x2 = min(image.shape[1] - 1, x + w + growth)
            y2 = min(image.shape[0] - 1, y + h + growth)
            region = image_grey[y1:y2, x1:x2]
            region = cv.normalize(region, None, 0, 255, cv.NORM_MINMAX)
            _, binary = cv.threshold(region, 0, 1, cv.THRESH_OTSU)
            binary = cv.medianBlur(binary, 3)
            border_mask = numpy.zeros_like(region, numpy.bool)
            border_mask[0, :] = True
            border_mask[-1, :] = True
            border_mask[:, 0] = True
            border_mask[:, -1] = True
            border = binary[border_mask]
            if numpy.mean(border) < 0.01:
                ok = True
                break
        if not ok:
            continue

        adjusted_boxes.append((x1, y1, x2 - x1, y2 - y1))

    # Find all the groups of boxes that are effectively the same, and from those groups choose only
    # the regions with minimal area.
    adjusted_boxes = list(set(adjusted_boxes))
    centres = [(x + w / 2, y + h / 2) for x, y, w, h in adjusted_boxes]
    equivalencies = []
    for i, (box1, (c1x, c1y)) in enumerate(zip(adjusted_boxes, centres)):
        tmp = {box1}
        for j, (box2, (c2x, c2y)) in enumerate(zip(adjusted_boxes, centres)):
            if box1 is not box2:
                # Consider boxes to be the same if their centres are close.
                distance = hypot(c1x - c2x, c1y - c2y)
                if distance < 5:
                    tmp.add(box2)
        equivalencies.append(tmp)
    for _ in equivalencies:
        for e1 in equivalencies:
            for e2 in equivalencies:
                if e1 != e2:
                    if any(b in e1 for b in e2):
                        for b in e2:
                            e1.add(b)
    adjusted_boxes = set(min(e, key=lambda b: b[2] * b[3]) for e in equivalencies)

    return adjusted_boxes


# Selects the regions (from detect_regions()) that form the house number.
def select_number(image, boxes):
    if len(boxes) <= 1:
        return boxes

    # TODO: drop lines with any boxes that differ in colour significantly

    # Find all possible lines with endpoints at box centres.
    lines = []
    for i, box1 in enumerate(boxes):
        c1x = box1[0] + box1[2] / 2
        c1y = box1[1] + box1[3] / 2
        for j, box2 in enumerate(boxes):
            if i != j:
                c2x = box2[0] + box2[2] / 2
                c2y = box2[1] + box2[3] / 2
                lines.append(((c1x, c1y), (c2x, c2y)))
    line_count = len(lines)
    lines = numpy.array(lines)

    # Associate each line with a set of boxes that intersect it.
    intersecting_boxes = []
    for line in lines:
        tmp = []
        for box in boxes:
            x1, y1, w, h = box
            x2 = x1 + w
            y2 = y1 + h
            box_lines = [((x1, y1), (x2, y1)),
                         ((x1, y1), (x1, y2)),
                         ((x1, y2), (x2, y2)),
                         ((x2, y1), (x2, y2))]
            if any(_lines_intersect(l, line) for l in box_lines):
                tmp.append(box)
        # Sort boxes by x coordinate, since digits run from left to right.
        tmp = sorted(tmp, key=lambda b: b[0])
        intersecting_boxes.append(tmp)

    box_counts = [len(bs) for bs in intersecting_boxes]

    # Compute various statistics on the lines' boxes.
    heights = [numpy.array([b[3] for b in bs]) for bs in intersecting_boxes]
    # Work with the median height rather than the mean height, so outliers have less impact and can
    # be detected more easily.
    median_heights = numpy.array([numpy.median(hs) for hs in heights])
    height_variations = [numpy.abs(heights[i] - median_heights[i]) for i in range(line_count)]
    # Find any lines that have at least 1 box with large variation in height from the others.
    big_height_variation = numpy.array(
        [any(height_variations[i] / median_heights[i] > 0.25) for i in range(line_count)])

    line_angles = numpy.arctan2(lines[:, 1, 1] - lines[:, 0, 1], lines[:, 1, 0] - lines[:, 0, 0])

    line_filter = numpy.zeros(len(lines), numpy.bool)
    # Select only roughly horizontal lines.
    line_filter[numpy.abs(line_angles) > radians(20)] = True
    # Remove lines whose boxes have large height variation.
    line_filter[big_height_variation] = True
    # Select the line with the tallest boxes (assume the house number digits are always bigger than
    # any other text).
    line_filter[median_heights / numpy.ma.max(numpy.ma.masked_array(median_heights, line_filter)) < 0.75] = True
    # Select the line with the most boxes.
    line_filter[box_counts < numpy.ma.max(numpy.ma.masked_array(box_counts, line_filter))] = True
    if numpy.all(line_filter):
        return None
    else:
        best_line = numpy.argmin(line_filter)
        return intersecting_boxes[best_line]


# Checks if 2 line segments (defined by endpoints) intersect.
def _lines_intersect(line1, line2):
    # Solve for line parameters s, t for intersection.
    d1 = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1])
    d2 = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1])
    A = [[d1[0], -d2[0]],
         [d1[1], -d2[1]]]
    b = [line2[0][0] - line1[0][0], line2[0][1] - line1[0][1]]
    try:
        s, t = numpy.linalg.solve(A, b)
    except numpy.linalg.LinAlgError:
        # No solution, no intersection.
        return False
    else:
        # Intersection if solution lies within the line segments
        return 0 <= s <= 1 and 0 <= t <= 1
