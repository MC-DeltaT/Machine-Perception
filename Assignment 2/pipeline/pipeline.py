from .digit_descriptor import digit_descriptor
from .number_extract import detect_regions, select_number
from .recognition_model import DigitRecognitionModel
from dataclasses import dataclass
import numpy
from typing import Sequence, Tuple


__all__ = [
    'HouseNumberRecognitionPipeline'
]


BoundingBox = Tuple[float, float, float, float]


class HouseNumberRecognitionPipeline:
    @dataclass
    class Result:
        plausible_boxes: Sequence[BoundingBox]
        house_number_boxes: Sequence[BoundingBox]
        house_number: Sequence[int]

    def __init__(self, recognition_model_file) -> None:
        self.recognition_model = DigitRecognitionModel(recognition_model_file)

    def process(self, image: numpy.ndarray) -> Result:
        plausible_boxes = detect_regions(image)
        house_number_boxes = select_number(image, plausible_boxes)
        house_number = []
        for x, y, w, h in house_number_boxes:
            region = image[y:y + h, x:x + w]
            descriptor = digit_descriptor(region)
            digit = self.recognition_model.predict(descriptor)
            house_number.append(digit)
        return self.Result(plausible_boxes, house_number_boxes, house_number)
