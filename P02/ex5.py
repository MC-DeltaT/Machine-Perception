import cv2 as cv


INPUT_FILE = 'data/prac02ex05img01.png'
OUTPUT_FILE = 'output/prac02ex05img01.png'

image = cv.imread(INPUT_FILE, cv.IMREAD_GRAYSCALE)
output = cv.equalizeHist(image)
cv.imwrite(OUTPUT_FILE, output)
