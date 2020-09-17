# Generates scaled versions of the sample images, used for task 1 and 2.

import os.path
from pathlib import Path
import cv2 as cv

IMAGES = [
    ('data/card.png', ((0, 0), (-1, -1))),
    ('data/dugong.jpg', ((360, 195), (460, 295)))
]
SCALES = [1, 1.1, 1.25, 1.5, 1.75, 2]
OUTPUT_DIR = 'generated/{}/scaled'
OUTPUT_FILE = '{}x.png'

for file, roi in IMAGES:
    image = cv.imread(file, cv.IMREAD_COLOR)
    filename = os.path.splitext(os.path.basename(file))[0]
    output_dir = OUTPUT_DIR.format(filename)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    x1, y1 = roi[0]
    x2, y2 = roi[1]
    image = image[y1:y2, x1:x2]
    for scale in SCALES:
        result = cv.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
        output_file = os.path.join(output_dir, OUTPUT_FILE.format(scale))
        cv.imwrite(output_file, result)
