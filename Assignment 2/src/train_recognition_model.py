# Trains the SVM used for digit recognition.

from pathlib import Path
from pipeline import DigitRecognitionModel
from recognition_training_data import load_recognition_training_data
from sys import argv


if len(argv) != 3:
    print('Usage: python3 train_recognition_model.py <dataset_dir> <output_file>')
    exit(1)

dataset_dir = Path(argv[1])
output_file = Path(argv[2])

inputs, labels = load_recognition_training_data(dataset_dir)

model = DigitRecognitionModel()
model.train(inputs, labels)
output_file.parent.mkdir(parents=True, exist_ok=True)
model.save(output_file)
