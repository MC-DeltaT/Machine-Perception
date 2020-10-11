import cv2 as cv
from glob import glob
import os.path
from pathlib import Path
from region_detect import detect_regions

OUTPUT_DIR = 'output/regions'

output_id = 0
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
for input_file in glob('train/*') + glob('val/*'):
    image = cv.imread(input_file, cv.IMREAD_COLOR)
    image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    boxes = detect_regions(image)
    for x1, y1, x2, y2 in boxes:
        region = image_grey[y1:y2, x1:x2]
        _, region = cv.threshold(region, 0, 255, cv.THRESH_OTSU + cv.THRESH_TOZERO)
        region = cv.resize(region, (20, 20), interpolation=cv.INTER_CUBIC)
        output_file = os.path.join(OUTPUT_DIR, f'{output_id}.png')
        cv.imwrite(output_file, region)
        output_id += 1
