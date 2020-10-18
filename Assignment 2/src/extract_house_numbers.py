import cv2 as cv
from os import listdir
from pathlib import Path
from pipeline import HouseNumberRecognitionPipeline
from sys import argv


NUMBER_TEXT_OUTPUT_FILE = 'House-{}.txt'
NUMBER_IMAGE_OUTPUT_FILE = 'DetectedArea-{}.jpg'
NUMBER_BOX_OUTPUT_FILE = 'BoundingBox-{}.txt'
RAW_BOXES_OUTPUT_FILE = 'RawBoxes-{}.png'
FILTERED_BOXES_OUTPUT_FILE = 'FilteredBoxes-{}.png'
IMAGE_EXTENSIONS = ('.jpg', '.png')


if len(argv) not in (4, 5):
    print('Usage: python3 extract_house_numbers.py <input_dir> <recognition_model_file> <output_dir> [<extra_output>]')
    exit(1)

input_dir = Path(argv[1])
recognition_model_file = argv[2]
output_dir = Path(argv[3])
extra_output = argv[4] == 'T' if len(argv) == 5 else False

pipeline = HouseNumberRecognitionPipeline(recognition_model_file)
output_dir.mkdir(parents=True, exist_ok=True)

for entry in listdir(input_dir):
    file_path = input_dir / Path(entry)
    filename = file_path.stem
    if file_path.suffix not in IMAGE_EXTENSIONS:
        continue
    image = cv.imread(str(file_path), cv.IMREAD_COLOR)
    image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    result = pipeline.process(image_grey)

    if extra_output:
        output = image.copy()
        for x, y, w, h in result.raw_regions:
            output = cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
        output_file = output_dir / RAW_BOXES_OUTPUT_FILE.format(filename)
        cv.imwrite(str(output_file), output)

        output = image.copy()
        for x, y, w, h in result.filtered_regions:
            output = cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
        output_file = output_dir / FILTERED_BOXES_OUTPUT_FILE.format(filename)
        cv.imwrite(str(output_file), output)

    if not result.house_number:
        print(f'No house number detected for {filename}')
        continue

    x1 = min(b[0] for b in result.house_number_regions)
    x2 = max(b[0] + b[2] for b in result.house_number_regions)
    y1 = min(b[1] for b in result.house_number_regions)
    y2 = max(b[1] + b[3] for b in result.house_number_regions)
    width = x2 - x1
    height = y2 - y1

    output_file = output_dir / NUMBER_BOX_OUTPUT_FILE.format(filename)
    with open(output_file, 'w') as file:
        file.write(f'{x1}, {y1}, {width}, {height}\n')

    output_file = output_dir / NUMBER_IMAGE_OUTPUT_FILE.format(filename)
    number_region = image[y1:y2, x1:x2]
    cv.imwrite(str(output_file), number_region)

    output_file = output_dir / NUMBER_TEXT_OUTPUT_FILE.format(filename)
    house_number = ''.join(map(str, result.house_number))
    with open(output_file, 'w') as file:
        file.write(f'Building {house_number}\n')
