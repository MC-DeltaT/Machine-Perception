# Evaluates the performance of the digit recognition model.

import numpy
from pathlib import Path
from pipeline import DigitRecognitionModel
from random import randrange
from recognition_training_data import load_recognition_training_data
from sys import argv


CROSS_VALIDATION_SETS = 4


if len(argv) != 2:
    print('Usage: python3 eval_recognition_model.py <dataset_dir>')
    exit(1)

dataset_dir = Path(argv[1])

inputs, labels = load_recognition_training_data(dataset_dir)

random_seed = randrange(2 ** 32)
numpy.random.default_rng(random_seed).shuffle(inputs)
numpy.random.default_rng(random_seed).shuffle(labels)

input_sets = numpy.array_split(inputs, CROSS_VALIDATION_SETS)
label_sets = numpy.array_split(labels, CROSS_VALIDATION_SETS)

for i in range(CROSS_VALIDATION_SETS):
    training_inputs = numpy.concatenate(numpy.delete(input_sets, i, 0))
    training_labels = numpy.concatenate(numpy.delete(label_sets, i, 0))
    test_inputs = input_sets[i]
    test_labels = label_sets[i]

    model = DigitRecognitionModel()
    model.train(training_inputs, training_labels)

    print(f'Round {i}')
    for name, inputs, labels in [('Training', training_inputs, training_labels), ('Test', test_inputs, test_labels)]:
        predictions = model.predict_multiple(inputs)
        accuracy = numpy.sum(predictions == labels) / inputs.shape[0]
        print(f'\t{name}: {round(accuracy * 100, 2)}% accuracy')
    print()
