import cv2 as cv


INPUT_FILE = 'data/prac02ex01img01.jpg'
OUTPUT_FILE = 'output/prac02ex01img01-{}.jpg'


image = cv.imread(INPUT_FILE, cv.IMREAD_COLOR)

conversions = [
    (cv.COLOR_BGR2GRAY, 'greyscale'),
    (cv.COLOR_BGR2HSV, 'hsv'),
    (cv.COLOR_BGR2LUV, 'luv'),
    (cv.COLOR_BGR2LAB, 'lab')
]

for code, name in conversions:
    result = cv.cvtColor(image, code)
    cv.imwrite(OUTPUT_FILE.format(name), result)
