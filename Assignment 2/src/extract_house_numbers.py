import cv2 as cv
from os import listdir
from pathlib import Path
from pipeline import HouseNumberRecognitionPipeline
from sys import argv


NUMBER_TEXT_OUTPUT_FILE = 'House-{}.txt'
NUMBER_IMAGE_OUTPUT_FILE = 'DetectedArea-{}.jpg'
BOUNDING_BOX_OUTPUT_FILE = 'BoundingBox-{}.txt'
IMAGE_EXTENSIONS = ('.jpg', '.png')


if len(argv) != 4:
    print('Usage: python3 extract_house_numbers.py <input_dir> <recognition_model_file> <output_dir>')
    exit(1)

input_dir = Path(argv[1])
recognition_model_file = argv[2]
output_dir = Path(argv[3])

pipeline = HouseNumberRecognitionPipeline(recognition_model_file)
output_dir.mkdir(parents=True, exist_ok=True)

for entry in listdir(input_dir):
    file_path = input_dir / Path(entry)
    filename = file_path.stem
    if file_path.suffix not in IMAGE_EXTENSIONS:
        continue
    image = cv.imread(str(file_path), cv.IMREAD_COLOR)

    result = pipeline.process(image)

    if not result.house_number:
        print(f'No house number detected for {filename}')
        continue

    x1 = min(b[0] for b in result.house_number_boxes)
    x2 = max(b[0] + b[2] for b in result.house_number_boxes)
    y1 = min(b[1] for b in result.house_number_boxes)
    y2 = max(b[1] + b[3] for b in result.house_number_boxes)
    width = x2 - x1
    height = y2 - y1

    output_file = output_dir / BOUNDING_BOX_OUTPUT_FILE.format(filename)
    with open(output_file, 'w') as file:
        file.write(f'{x1}, {y1}, {width}, {height}')

    output_file = output_dir / NUMBER_IMAGE_OUTPUT_FILE.format(filename)
    number_region = image[y1:y2, x1:x2]
    cv.imwrite(str(output_file), number_region)

    output_file = output_dir / NUMBER_TEXT_OUTPUT_FILE.format(filename)
    house_number = ''.join(map(str, result.house_number))
    with open(output_file, 'w') as file:
        file.write('Building ' + house_number)
