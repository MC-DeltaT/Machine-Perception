import cv2 as cv
import numpy


INPUT_FILE = 'data/prac04ex02img01.png'
OUTPUT_FILE = 'output/prac04ex02img01-boxes.png'

image = cv.imread(INPUT_FILE, cv.IMREAD_GRAYSCALE)
height, width = image.shape
_, binary = cv.threshold(image, 0, 1, cv.THRESH_OTSU)
binary = 1 - binary
labels = numpy.zeros_like(binary, dtype=numpy.uint8)

label = 1
queue = []
for x, y in numpy.ndindex(width, height):
    if labels[y, x] == 0 and binary[y, x]:
        labels[y, x] = label
        queue.append((x, y))
        while queue:
            x, y = queue.pop(0)
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                x_ = x + dx
                y_ = y + dy
                if 0 <= x_ < width and 0 <= y_ < height:
                    if labels[y_, x_] == 0 and binary[y_, x_]:
                        labels[y_, x_] = label
                        queue.append((x_, y_))
        label += 1

region_count = numpy.max(labels)
regions = [[] for _ in range(region_count)]
for x, y in numpy.ndindex(width, height):
    label = labels[y, x]
    if label > 0:
        regions[label - 1].append((x, y))

boxes = [[] for _ in range(region_count)]
for r in range(len(regions)):
    min_x = width - 1
    min_y = height - 1
    max_x = 0
    max_y = 0
    for x, y in regions[r]:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    boxes[r] = ((min_x, min_y), (max_x, max_y))

areas = [len(r) for r in regions]
box_widths = [b[1][0] - b[0][0] + 1 for b in boxes]
box_heights = [b[1][1] - b[0][1] + 1 for b in boxes]
fg_ratios = [areas[i] / (box_widths[i] * box_heights[i]) for i in range(region_count)]

output = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
for point1, point2 in boxes:
    output = cv.rectangle(output, point1, point2, (0, 0, 255))
cv.imwrite(OUTPUT_FILE, output)

output = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
for point1, point2 in boxes:
    output = cv.rectangle(output, point1, point2, (0, 0, 255))
cv.imwrite(OUTPUT_FILE, output)
