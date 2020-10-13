import cv2 as cv
from math import hypot, radians
import numpy


# Finds regions that are possibly digits.
def detect_regions(image):
    image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_grey = cv.normalize(image_grey, None, 0, 255, cv.NORM_MINMAX)

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

        # Try to grow region until it covers entire foreground area. If this can't be done, assume
        # it's not a digit.
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

    # TODO: refactor
    # TODO: drop lines with any boxes that differ in height significantly
    # TODO: drop lines with any boxes that differ in colour significantly
    # TODO: sort boxes by x

    # Find all possible lines with endpoints at box centres.
    lines = numpy.zeros((len(boxes), len(boxes), 2, 2), numpy.float)
    for i, box1 in enumerate(boxes):
        c1x = box1[0] + box1[2] / 2
        c1y = box1[1] + box1[3] / 2
        for j, box2 in enumerate(boxes):
            if i != j:
                c2x = box2[0] + box2[2] / 2
                c2y = box2[1] + box2[3] / 2
                lines[i, j] = [[c1x, c1y], [c2x, c2y]]

    # Associate each line with a set of boxes that intersect it.
    intersecting_boxes = numpy.empty((len(boxes), len(boxes)), numpy.object)
    box_counts = numpy.zeros((len(boxes), len(boxes)), numpy.uint16)
    for i, j in numpy.ndindex(len(boxes), len(boxes)):
        tmp = []
        if i != j:
            line = lines[i, j]
            for box in boxes:
                x1, y1, w, h = box
                x2 = x1 + w
                y2 = y1 + h
                box_lines = [((x1, y1), (x2, y1)),
                             ((x1, y1), (x1, y2)),
                             ((x1, y2), (x2, y2)),
                             ((x2, y1), (x2, y2))]
                if any(lines_intersect(l, line) for l in box_lines):
                    tmp.append(box)
        intersecting_boxes[i, j] = tmp
        box_counts[i, j] = len(tmp)

    # Compute various statistics on the lines' boxes.
    heights = numpy.empty((len(boxes), len(boxes)), numpy.object)
    height_variations = numpy.zeros((len(boxes), len(boxes)), numpy.float)
    average_heights = numpy.zeros((len(boxes), len(boxes)), numpy.float)
    for i, j in numpy.ndindex(len(boxes), len(boxes)):
        if i != j:
            hs = numpy.array([b[3] for b in intersecting_boxes[i, j]])
            avg_h = numpy.mean(hs)
            height_variations[i, j] = numpy.mean(numpy.abs(hs - avg_h))
            heights[i, j] = hs
            average_heights[i, j] = avg_h
    line_angles = numpy.arctan2(lines[:, :, 1, 1] - lines[:, :, 0, 1], lines[:, :, 1, 0] - lines[:, :, 0, 0])

    line_filter = numpy.zeros((len(boxes), len(boxes)), numpy.bool)
    numpy.fill_diagonal(line_filter, True)
    # Select only roughly horizontal lines.
    line_filter[numpy.abs(line_angles) > radians(20)] = True
    # Remove lines whose boxes have large height variation.
    line_filter[height_variations > average_heights * 0.2] = True
    # Select the line with the tallest boxes (assume the house number digits are always bigger than
    # any other text).
    line_filter[average_heights / numpy.ma.max(numpy.ma.masked_array(average_heights, line_filter)) < 0.8] = True
    # Select the line with the most boxes.
    line_filter[box_counts < numpy.ma.max(numpy.ma.masked_array(box_counts, line_filter))] = True
    if numpy.all(line_filter):
        return None
    else:
        best_line = numpy.unravel_index(numpy.argmin(line_filter), line_filter.shape)
        return intersecting_boxes[best_line]


# Checks if 2 line segments (defined by endpoints) intersect.
def lines_intersect(line1, line2):
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
