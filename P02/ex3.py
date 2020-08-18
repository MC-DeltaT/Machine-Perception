import cv2 as cv


INPUT_FILE = 'data/prac02ex03img01.jpg'
OUTPUT_FILE = 'output/prac02ex03img01-median_{}.jpg'
KERNEL_RADII = [3, 5, 7, 9]

image = cv.imread(INPUT_FILE, cv.IMREAD_GRAYSCALE)

for radius in KERNEL_RADII:
    result = cv.medianBlur(image, radius)
    cv.imwrite(OUTPUT_FILE.format(radius), result)
