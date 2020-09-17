# Generates rotated versions of the sample images, used for task 1 and 2.

import os.path
from pathlib import Path
import cv2 as cv

INPUT = [
    ('data/card.png', ((51, 20), (131, 100))),
    ('data/dugong.jpg', ((360, 195), (460, 295)))
]
ROTATIONS = [0, 15, 45, 75]
OUTPUT_DIR = 'generated/{}/rotated'
OUTPUT_FILE = '{}deg.png'

for file, roi in INPUT:
    image = cv.imread(file, cv.IMREAD_COLOR)
    filename = os.path.splitext(os.path.basename(file))[0]
    output_dir = OUTPUT_DIR.format(filename)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    x1, y1 = roi[0]
    x2, y2 = roi[1]
    centre = ((x1 + x2) / 2, (y1 + y2) / 2)
    for angle in ROTATIONS:
        height, width, _ = image.shape
        rotation = cv.getRotationMatrix2D(centre, angle, 1)
        result = cv.warpAffine(image, rotation, (width, height), flags=cv.INTER_CUBIC)
        result = result[y1:y2, x1:x2]
        output_file = os.path.join(output_dir, OUTPUT_FILE.format(angle))
        cv.imwrite(output_file, result)
