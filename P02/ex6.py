import glob
import os.path

import cv2 as cv


INPUT_FILE = 'data/prac02ex06img*.png'
OUTPUT_FILE = 'output/{}-{}_{}.png'
OPERATIONS = [
    ('dilate', cv.dilate),
    ('erode', cv.erode),
    ('open', lambda im, s: cv.dilate(cv.erode(im, s), s)),
    ('close', lambda im, s: cv.erode(cv.dilate(im, s), s)),
    ('morphgrad', lambda im, s: cv.dilate(im, s) - cv.erode(im, s)),
    ('blackhat', lambda im, s: cv.erode(cv.dilate(im, s), s) - im)
]
S_SIZES = [3, 5, 7]

for file in glob.glob(INPUT_FILE):
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    for op_name, op in OPERATIONS:
        for size in S_SIZES:
            s = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
            output = op(image, s)
            filename = os.path.splitext(os.path.basename(file))[0]
            output_file = OUTPUT_FILE.format(filename, op_name, size)
            cv.imwrite(output_file, output)
