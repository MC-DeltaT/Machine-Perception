from .digit_descriptor import digit_descriptor
from .number_extract import detect_regions, filter_regions, select_number
from .recognition_model import DigitRecognitionModel
import numpy
from typing import Sequence, Tuple


__all__ = [
    'HouseNumberRecognitionPipeline'
]


BoundingBox = Tuple[float, float, float, float]


class HouseNumberRecognitionPipeline:
    class Result:
        def __init__(self, raw_regions: Sequence[BoundingBox],
                     filtered_regions: Sequence[BoundingBox],
                     house_number_regions: Sequence[BoundingBox],
                     house_number: Sequence[int]) -> None:
            self.raw_regions = raw_regions
            self.filtered_regions = filtered_regions
            self.house_number_regions = house_number_regions
            self.house_number = house_number

    def __init__(self, recognition_model_file) -> None:
        self.recognition_model = DigitRecognitionModel(recognition_model_file)

    def process(self, image_grey: numpy.ndarray) -> Result:
        raw_boxes, regions = detect_regions(image_grey)
        filtered_boxes = filter_regions(image_grey, raw_boxes, regions)
        house_number_boxes = select_number(filtered_boxes)
        house_number = []
        for x, y, w, h in house_number_boxes:
            region = image_grey[y:y + h, x:x + w]
            descriptor = digit_descriptor(region)
            digit = self.recognition_model.predict_single(descriptor)
            house_number.append(digit)
        return self.Result(raw_boxes, filtered_boxes, house_number_boxes, house_number)
