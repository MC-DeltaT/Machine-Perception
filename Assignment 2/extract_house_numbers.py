import cv2 as cv
from digit_descriptor import digit_descriptor
from number_extract import detect_regions, select_number
import numpy
from os import listdir
from pathlib import Path
from sys import argv


NUMBER_TEXT_OUTPUT_FILE = 'House-{}.txt'
NUMBER_IMAGE_OUTPUT_FILE = 'DetectedArea-{}.jpg'
BOUNDING_BOX_OUTPUT_FILE = 'BoundingBox-{}.txt'
REGION_BOX_OUTPUT_FILE = 'RegionBoxes-{}.png'
IMAGE_EXTENSIONS = ('.jpg', '.png')
MAX_IMAGE_SIZE = 500
OUTPUT_REGION_BOXES = False


if len(argv) != 4:
    print('Usage: python3 extract_house_numbers.py <input_dir> <recognition_model_file> <output_dir>')
    exit(1)

input_dir = Path(argv[1])
recognition_model_file = argv[2]
output_dir = Path(argv[3])

output_dir.mkdir(parents=True, exist_ok=True)
recognition_model = cv.ml.SVM_load(recognition_model_file)

for entry in listdir(input_dir):
    file_path = input_dir / Path(entry)
    filename = file_path.stem
    if file_path.suffix not in IMAGE_EXTENSIONS:
        continue
    image = cv.imread(str(file_path), cv.IMREAD_COLOR)

    # Downscale large images for reliable MSER detection and performance reasons.
    max_dim = max(image.shape[0], image.shape[1])
    if max_dim > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max_dim
        image = cv.resize(image, None, fx=scale, fy=scale)

    boxes = detect_regions(image)

    # Output detected regions for debugging.
    if OUTPUT_REGION_BOXES:
        result = image.copy()
        for x, y, w, h in boxes:
            result = cv.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 1)
        output_file = output_dir / REGION_BOX_OUTPUT_FILE.format(filename)
        cv.imwrite(str(output_file), result)

    boxes = select_number(image, boxes)

    if not boxes:
        print(f'No house number detected for {filename}')
        continue

    digits = []
    for x, y, w, h in boxes:
        region = image[y:y + h, x:x + w]
        descriptor = digit_descriptor(region)
        digit = int(recognition_model.predict(numpy.array([descriptor]))[1][0][0])
        digits.append(digit)

    x1 = min(b[0] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y1 = min(b[1] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    width = x2 - x1
    height = y2 - y1

    output_file = output_dir / BOUNDING_BOX_OUTPUT_FILE.format(filename)
    with open(output_file, 'w') as file:
        file.write(f'{x1}, {y1}, {width}, {height}')

    output_file = output_dir / NUMBER_IMAGE_OUTPUT_FILE.format(filename)
    number_region = image[y1:y2, x1:x2]
    cv.imwrite(str(output_file), number_region)

    output_file = output_dir / NUMBER_TEXT_OUTPUT_FILE.format(filename)
    with open(output_file, 'w') as file:
        file.write('Building ' + ''.join(map(str, digits)))
