# Visualises the comparison of SIFT and HOG descriptors between transformed versions of the images.
# gen_scaled.py and gen_rotated.py should be run before running this script.

from collections import defaultdict
import os.path
from pathlib import Path
import cv2 as cv
from matplotlib import pyplot
import numpy

def inverse_scale(image_size, scale, point):
    T = [[1 / scale, 0],
         [0, 1 / scale]]
    return numpy.dot(point, T)

def inverse_rotation(image_size, angle, point):
    centre = (image_size[0] / 2, image_size[1] / 2)
    T = cv.getRotationMatrix2D(centre, -angle, 1)
    return numpy.dot(T, [point[0], point[1], 1])

BASE_IMAGES = ['card', 'dugong']
TRANSFORMS = [
    ('scaled', '{}x.png', [1, 1.25, 1.5, 2], inverse_scale, 'Scale amount'),
    ('rotated', '{}deg.png', [0, 15, 45, 75], inverse_rotation, 'Rotation angle')
]
INPUT_DIR = 'generated/{}/{}'
OUTPUT_DIR = 'results/{}/{}'
KEYPOINT_MATCH_OUTPUT = 'keypoint_matches.png'
SIFT_COMPARE_OUTPUT = 'sift_descriptor_compare.png'
KEYPOINT_MATCH_THRESHOLD = 1

sift = cv.xfeatures2d.SIFT_create()
for base_image in BASE_IMAGES:
    for transform_name, file_template, transform_vals, inverse_transform, transform_axis_label in TRANSFORMS:
        # METHODOLOGY:
        # To choose a keypoint to examine, we first get all keypoints from all versions of the image.
        # We then apply to the keypoints the inverse of the transformation that was applied to that image, to get the
        # locations where the keypoints "should be" if SIFT was perfectly scale/rotation invariant.
        # Finally, we select keypoints that match in location across all versions of the image.

        input_dir = INPUT_DIR.format(base_image, transform_name)
        images = []
        keypoints = []
        for transform_val in transform_vals:
            file = os.path.join(input_dir, file_template.format(transform_val))
            image = cv.imread(file, cv.IMREAD_COLOR)
            images.append(image)
            actual_keypoints = sift.detect(image, None)
            adjusted_keypoints = [inverse_transform((image.shape[1], image.shape[0]), transform_val, kp.pt)
                                  for kp in actual_keypoints]
            keypoints.append(list(zip(actual_keypoints, adjusted_keypoints)))

        # Find keypoints in the transformed images that match in location ain the original image.
        matches = defaultdict(list)
        for i in range(len(transform_vals) - 1):
            for keypoint1, _ in keypoints[0]:
                ms = [keypoint2 for keypoint2, adjusted_point in keypoints[i]
                      if numpy.linalg.norm(keypoint1.pt - adjusted_point) <= KEYPOINT_MATCH_THRESHOLD]
                if ms:
                    # Only need one match, so choose the strongest keypoint.
                    match = max(ms, key=lambda kp: kp.response)
                    matches[keypoint1].append(match)

        # Select only keypoints that match in all the image versions.
        matches = {kp: m for kp, m in matches.items() if len(m) == len(transform_vals) - 1}

        output_dir = OUTPUT_DIR.format(base_image, transform_name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        result = cv.drawKeypoints(images[0].copy(), list(matches), None)
        output_file = os.path.join(output_dir, KEYPOINT_MATCH_OUTPUT)
        cv.imwrite(output_file, result)

        # Choose the strongest keypoint to examine.
        target_keypoint = max(matches, key=lambda kp: kp.response)
        target_keypoints = [target_keypoint]
        target_keypoints.extend(matches[target_keypoint])

        # Compare the keypoint SIFT descriptors to the descriptor for the original image, using Euclidean distance.
        descriptors = [sift.compute(image, [kp])[1] for image, kp in zip(images, target_keypoints)]
        distances = [numpy.linalg.norm(desc - descriptors[0]) for desc in descriptors]

        pyplot.clf()
        pyplot.scatter(transform_vals, distances)
        pyplot.xlabel(transform_axis_label)
        pyplot.ylabel('Distance from original')
        pyplot.tight_layout()
        output_file = os.path.join(output_dir, SIFT_COMPARE_OUTPUT)
        pyplot.savefig(output_file)

        # TODO: compare keypoint across transformations
