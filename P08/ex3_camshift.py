import cv2 as cv
import numpy
from pathlib import Path


INPUT_FILES = [
    'data/signs/sign01.png',
    'data/signs/sign02.png',
    'data/signs/sign03.png',
    'data/signs/sign04.png',
    'data/signs/sign05.png'
]

rect = (630, 330, 130, 120)
image = cv.imread('data/signs/sign01.png', cv.IMREAD_COLOR)
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
roi = hsv[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
roi_hist = cv.calcHist([roi], [0], None, [256], (0, 256))
roi_hist = cv.normalize(roi_hist, None, 0, 255, cv.NORM_MINMAX)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1)
Path('output').mkdir(parents=True, exist_ok=True)
for file in INPUT_FILES:
    file = Path(file)
    image = cv.imread(str(file), cv.IMREAD_COLOR)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    backprojection = cv.calcBackProject([hsv], [0], roi_hist, (0, 180), 1)
    area, rect = cv.CamShift(backprojection, rect, criteria)

    points = cv.boxPoints(area).astype(numpy.int32)
    result = image.copy()
    result = cv.polylines(result, [points], True, (0, 255, 0), 1)
    output_file = Path('output') / f'{file.stem}-camshift.png'
    cv.imwrite(str(output_file), result)
