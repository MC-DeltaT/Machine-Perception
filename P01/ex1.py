import glob
import os.path

import cv2
from matplotlib import pyplot


INPUT_DIR = 'data'
OUTPUT_DIR = 'output'


for input_file in glob.glob(os.path.join(INPUT_DIR, f'prac01ex01img*.png')):
    filename = os.path.basename(input_file)
    name, ext = os.path.splitext(filename)
    print(input_file)
    image = cv2.imread(input_file)
    print(f'Width: {image.shape[1]}')
    print(f'Height: {image.shape[0]}')
    for channel, colour in enumerate(('blue', 'green', 'red')):
        histogram = cv2.calcHist([image], [channel], None, [10], [0, 256])
        pyplot.plot(histogram, color=colour)
    pyplot.title(name)
    pyplot.show()
    resized = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{name}-resized.png'), resized)
    print()
