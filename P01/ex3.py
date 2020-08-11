import cv2


CROP_FILE = 'data/prac01ex02crop.txt'
INPUT_FILE = 'data/prac01ex02img01.png'
OUTPUT_FILE = 'output/prac01ex02img01-marked.png'


with open(CROP_FILE) as file:
    x1, y1, x2, y2 = [int(x) for x in file.read().strip().split()]
print(f'Crop top left: ({x1}, {y1})')
print(f'Crop bottom right: ({x2}, {y2})')

image = cv2.imread(INPUT_FILE)
image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
for point in ((x1, y1), (x1, y2), (x2, y1), (x2, y2)):
    image = cv2.circle(image, point, 5, (0, 0, 255))
cv2.imwrite(OUTPUT_FILE, image)
