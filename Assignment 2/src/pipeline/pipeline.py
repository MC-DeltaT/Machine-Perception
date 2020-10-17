from .digit_descriptor import digit_descriptor
from .number_extract import detect_regions, select_number
from .recognition_model import DigitRecognitionModel
import numpy
from typing import Sequence, Tuple


__all__ = [
    'HouseNumberRecognitionPipeline'
]


BoundingBox = Tuple[float, float, float, float]


class HouseNumberRecognitionPipeline:
    class Result:
        def __init__(self, plausible_boxes: Sequence[BoundingBox],
                     house_number_boxes: Sequence[BoundingBox], house_number: Sequence[int]) -> None:
            self.plausible_boxes = plausible_boxes
            self.house_number_boxes = house_number_boxes
            self.house_number = house_number

    def __init__(self, recognition_model_file) -> None:
        self.recognition_model = DigitRecognitionModel(recognition_model_file)

    def process(self, image: numpy.ndarray) -> Result:
        plausible_boxes = detect_regions(image)
        house_number_boxes = select_number(image, plausible_boxes)
        house_number = []
        for x, y, w, h in house_number_boxes:
            region = image[y:y + h, x:x + w]
            descriptor = digit_descriptor(region)
            digit = self.recognition_model.predict_single(descriptor)
            house_number.append(digit)
        return self.Result(plausible_boxes, house_number_boxes, house_number)
