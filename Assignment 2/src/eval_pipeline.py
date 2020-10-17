# Evaluates the performance of the full pipeline.

import cv2 as cv
from os import listdir
from pathlib import Path
from pipeline import HouseNumberRecognitionPipeline
from sys import argv


IMAGE_EXTENSIONS = ('.jpg', '.png')


if len(argv) != 4:
    print('Usage: python3 eval_pipeline.py <input_dir> <recognition_model_file> <correct_numbers_file>')
    exit(1)

input_dir = Path(argv[1])
recognition_model_file = argv[2]
correct_numbers_file = Path(argv[3])

pipeline = HouseNumberRecognitionPipeline(recognition_model_file)

with open(correct_numbers_file) as file:
    correct_numbers = eval(file.read())

results = []
for entry in listdir(input_dir):
    file_path = input_dir / Path(entry)
    filename = file_path.stem
    if file_path.suffix not in IMAGE_EXTENSIONS:
        continue
    image = cv.imread(str(file_path), cv.IMREAD_GRAYSCALE)

    result = pipeline.process(image)
    house_number = ''.join(map(str, result.house_number))

    results.append((filename, house_number, correct_numbers[filename]))

correct = [r for r in results if r[1] == r[2]]
incorrect = [r for r in results if r[1] != r[2]]

total = len(correct) + len(incorrect)
print(f'Accuracy: {round(100 * len(correct) / total, 2)}%')

if correct:
    print()
    print('Correct:')
    for image_name, predicted_number, correct_number in correct:
        print(f'\t{image_name}: c={correct_number}, p={predicted_number}')

if incorrect:
    print()
    print('Incorrect:')
    for image_name, predicted_number, correct_number in incorrect:
        print(f'\t{image_name}: c={correct_number}, p={predicted_number}')
