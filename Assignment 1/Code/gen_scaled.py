import os.path
import cv2 as cv

IMAGES = [
    'data/card.png',
    'data/dugong.jpg'
]
SCALES = [1.5, 2, 2.5]
OUTPUT_FILE = 'results/{}-scaled_{}x.png'

for file in IMAGES:
    image = cv.imread(file, cv.IMREAD_COLOR)
    filename = os.path.splitext(os.path.basename(file))[0]

    for scale in SCALES:
        result = cv.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
        output_file = OUTPUT_FILE.format(filename, scale)
        cv.imwrite(output_file, result)
