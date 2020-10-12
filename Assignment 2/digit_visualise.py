import cv2 as cv
from glob import glob
import os.path
from pathlib import Path
from digit_detect import detect_regions, select_number

OUTPUT_DIR = 'output/number_detection'

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
for input_file in glob('data/full/train/*'):
    filename = os.path.splitext(os.path.basename(input_file))[0]
    image = cv.imread(input_file, cv.IMREAD_COLOR)
    image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    boxes = detect_regions(image)
    result = image.copy()
    for x, y, w, h in boxes:
        result = cv.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 1)
    boxes = select_number(image, boxes)
    for x, y, w, h in boxes:
        result = cv.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 1)
    output_file = os.path.join(OUTPUT_DIR, f'{filename}.png')
    cv.imwrite(output_file, result)
