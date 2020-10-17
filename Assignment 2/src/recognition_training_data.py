# Loads the digit recognition training data.

import cv2 as cv
from pipeline import digit_descriptor
import numpy
from os import listdir
from pathlib import Path
from typing import Tuple


__all__ = [
    'load_recognition_training_data'
]


# Loads the digit recognition training data provided from Blackboard.
def load_recognition_training_data(input_dir: Path) -> Tuple[numpy.ndarray, numpy.ndarray]:
    inputs = []
    labels = []
    for i in range(10):
        directory = input_dir / str(i)
        for entry in listdir(directory):
            file_path = directory / Path(entry)
            if file_path.suffix not in IMAGE_EXTENSIONS:
                continue
            image = cv.imread(str(file_path), cv.IMREAD_COLOR)
            descriptor = digit_descriptor(image)
            inputs.append(descriptor)
            labels.append(i)

    inputs = numpy.array(inputs, numpy.float32)
    labels = numpy.array(labels, numpy.int32)
    return inputs, labels


IMAGE_EXTENSIONS = ('.jpg', '.png')
