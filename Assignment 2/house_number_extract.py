import cv2 as cv
from digit_descriptor import digit_descriptor
from digit_detect import detect_regions, select_number
import numpy
from os import listdir
import os.path
from pathlib import Path
from sys import argv


NUMBER_TEXT_OUTPUT_FILE = 'House-{}.txt'
NUMBER_IMAGE_OUTPUT_FILE = 'DetectedArea-{}.jpg'
BOUNDING_BOX_OUTPUT_FILE = 'BoundingBox-{}.txt'
IMAGE_EXTENSIONS = ('.jpg', '.png')


if len(argv) != 4:
    print('Usage: python3 house_number_extract.py <input_dir> <recognition_model_file> <output_dir>')
    exit(1)

input_dir = argv[1]
recognition_model_file = argv[2]
output_dir = argv[3]

Path(output_dir).mkdir(parents=True, exist_ok=True)

recognition_model = cv.ml.SVM_load(recognition_model_file)
for entry in listdir(input_dir):
    filename, ext = os.path.splitext(entry)
    if ext not in IMAGE_EXTENSIONS:
        continue
    file_path = os.path.join(input_dir, entry)
    image = cv.imread(file_path, cv.IMREAD_COLOR)
    image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    boxes = detect_regions(image)
    boxes = select_number(image, boxes)
    digits = []
    for x, y, w, h in boxes:
        region = image_grey[y:y + h, x:x + w]
        descriptor = digit_descriptor(region)
        digit = recognition_model.predict(numpy.array([descriptor]))[1][0][0]
        digits.append(int(digit))

    x1 = min(b[0] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y1 = min(b[1] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    width = x2 - x1
    height = y2 - y1

    output_file = os.path.join(output_dir, BOUNDING_BOX_OUTPUT_FILE.format(filename))
    with open(output_file, 'w') as file:
        file.write(f'{x1}, {y1}, {width}, {height}')

    output_file = os.path.join(output_dir, NUMBER_IMAGE_OUTPUT_FILE.format(filename))
    number_region = image[y1:y2, x1:x2]
    cv.imwrite(output_file, number_region)

    output_file = os.path.join(output_dir, NUMBER_TEXT_OUTPUT_FILE.format(filename))
    with open(output_file, 'w') as file:
        file.write('Building ' + ''.join(map(str, digits)))
