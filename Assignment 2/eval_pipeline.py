# Evaluates the performance of the full pipeline.

from os import listdir
from pathlib import Path
import re
from sys import argv


if len(argv) != 3:
    print('Usage: python3 eval_pipeline.py <pipeline_output_dir> <correct_numbers_file>')
    exit(1)

pipeline_output_dir = Path(argv[1])
correct_numbers_file = Path(argv[2])

with open(correct_numbers_file) as file:
    correct_numbers = eval(file.read())

total = 0
incorrect = []
for entry in listdir(pipeline_output_dir):
    match = re.fullmatch('House-(.+)\\.txt', entry)
    if not match:
        continue
    file_path = pipeline_output_dir / entry
    with open(file_path) as file:
        predicted_number = file.read().rstrip()[9:]
    image_name = match.group(1)
    correct_number = correct_numbers[image_name]
    if predicted_number != correct_number:
        incorrect.append((image_name, predicted_number, correct_number))
    total += 1

correct = total - len(incorrect)
print(f'Accuracy: {round(100 * correct / total, 2)}%')

if incorrect:
    print()
    print('Incorrect:')
    for image_name, predicted_number, correct_number in incorrect:
        print(f'\t{image_name}: c={correct_number}, p={predicted_number}')
