import glob
import os.path

import cv2 as cv


INPUT_FILES = 'data/prac03ex02img*'
OUTPUT_FILE = 'output/{}-canny.png'

THRESHOLD1 = 200
THRESHOLD2 = 300

for file in glob.glob(INPUT_FILES):
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    result = cv.Canny(image, THRESHOLD1, THRESHOLD2, L2gradient=True)
    filename = os.path.splitext(os.path.basename(file))[0]
    output_file = OUTPUT_FILE.format(filename)
    cv.imwrite(output_file, result)
