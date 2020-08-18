import cv2 as cv


INPUT_FILE = 'data/prac02ex04img01.png'
OUTPUT_FILE = 'output/prac02ex04img01.png'

ALPHA = 1.1
BETA = 0

image = cv.imread(INPUT_FILE, cv.IMREAD_GRAYSCALE)
output = ALPHA * image + BETA
cv.imwrite(OUTPUT_FILE, output)
