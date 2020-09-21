# Visualises the comparison of SIFT and HOG descriptors between transformed versions of the images.
# gen_scaled.py and gen_rotated.py should be run before running this script.

import cv2 as cv
from matplotlib import pyplot
import numpy
import os.path
from pathlib import Path

def inverse_scale(image_size, scale, keypoint):
    T = [[1 / scale, 0],
         [0, 1 / scale]]
    return numpy.dot(keypoint.pt, T), keypoint.angle

def inverse_rotation(image_size, angle, keypoint):
    centre = (image_size[0] / 2, image_size[1] / 2)
    T = cv.getRotationMatrix2D(centre, -angle, 1)
    return numpy.dot(T, [keypoint.pt[0], keypoint.pt[1], 1]), keypoint.angle + angle

BASE_IMAGES = ['card', 'dugong']
TRANSFORMS = [
    ('scaled', '{}x.png', [1, 1.2, 1.4, 1.6, 1.8, 2], inverse_scale, 'Scale amount'),
    ('rotated', '{}deg.png', [0, 15, 30, 45, 60, 75], inverse_rotation, 'Rotation angle')
]
INPUT_DIR = 'generated/{}/{}'
OUTPUT_DIR = 'results/{}/{}'
KEYPOINT_MATCH_OUTPUT = 'keypoint_matches.png'
DESCRIPTOR_COMPARE_OUTPUT = 'descriptor_compare.png'
KEYPOINT_MATCH_DIST_THRESHOLD = 1
KEYPOINT_MATCH_ANGLE_THRESHOLD = 15

sift = cv.xfeatures2d.SIFT_create()
hog = cv.HOGDescriptor((16, 16), (16, 16), (8, 8), (8, 8), 9)
for base_image in BASE_IMAGES:
    for transform_name, file_template, transform_vals, inverse_transform, transform_axis_label in TRANSFORMS:
        # METHODOLOGY:
        # To choose keypoints to examine, we first get all keypoints from all versions of the image. We then apply to
        # the keypoints the inverse of the transformation that was applied to that image, to get the locations and
        # orientations the keypoints "should have" if SIFT was perfectly scale/rotation invariant. Finally, we select
        # keypoints that match in location and orientation across all versions of the image.

        input_dir = INPUT_DIR.format(base_image, transform_name)
        image_versions = len(transform_vals)
        images = []
        keypoints = []
        for transform_val in transform_vals:
            file = os.path.join(input_dir, file_template.format(transform_val))
            image = cv.imread(file, cv.IMREAD_COLOR)
            images.append(image)
            actual_keypoints = sift.detect(image, None)
            adjusted_keypoints = [inverse_transform((image.shape[1], image.shape[0]), transform_val, kp)
                                  for kp in actual_keypoints]
            keypoints.append(list(zip(actual_keypoints, adjusted_keypoints)))

        # Find keypoints in the transformed images that match in location and orientation with the original image.
        keypoint_matches = [[kp] for kp, _ in keypoints[0]]
        for i in range(1, image_versions):
            for j, (orig_keypoint, _) in enumerate(keypoints[0]):
                matches = [(kp, numpy.linalg.norm(orig_keypoint.pt - adjusted_kp[0]), abs(orig_keypoint.angle - adjusted_kp[1]))
                           for kp, adjusted_kp in keypoints[i]]
                matches = [(kp, dist, angle_diff) for kp, dist, angle_diff in matches
                           if dist < KEYPOINT_MATCH_DIST_THRESHOLD and angle_diff < KEYPOINT_MATCH_ANGLE_THRESHOLD]
                if matches:
                    # Only need one match, so consider the distance, angle difference, and keypoint response to
                    # determine which is the "best" match.
                    def match_cost(m):
                        return m[1] / KEYPOINT_MATCH_DIST_THRESHOLD + m[2] / KEYPOINT_MATCH_ANGLE_THRESHOLD - 10 * m[0].response
                    match, _, _ = min(matches, key=match_cost)
                    keypoint_matches[j].append(match)

        # Select only keypoints that match in all the image versions.
        keypoint_matches = [matches for matches in keypoint_matches if len(matches) == image_versions]
        keypoint_matches = numpy.transpose(keypoint_matches)

        output_dir = OUTPUT_DIR.format(base_image, transform_name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        result = cv.drawKeypoints(images[0].copy(), keypoint_matches[0], None)
        output_file = os.path.join(output_dir, KEYPOINT_MATCH_OUTPUT)
        cv.imwrite(output_file, result)

        sift_descriptors = numpy.array([sift.compute(image, keypoint_matches[i])[1] for i, image in enumerate(images)])
        # For some reason OpenCV makes the norm 512 instead of 1, so fix that so we can compare with HOG more easily.
        sift_descriptors /= 512

        hog_descriptors = numpy.empty((image_versions, keypoint_matches.shape[1], 36), dtype=numpy.float)
        for i, image in enumerate(images):
            for j, keypoint in enumerate(keypoint_matches[i]):
                # To get the HOG descriptor around the keypoint, we extract the 16x16 region around the keypoint, and
                # calculate HOG on that (easier than calculating HOG on the whole image then extracting the region.)
                # Need to clamp region to bounds of image (keypoint might be near edge of image).
                region_cx = numpy.clip(round(keypoint.pt[0]), 8, image.shape[1] - 8)
                region_cy = numpy.clip(round(keypoint.pt[1]), 8, image.shape[0] - 8)
                region = image[region_cy - 8:region_cy + 8, region_cx - 8:region_cx + 8]
                descriptor = hog.compute(region).ravel()
                hog_descriptors[i][j] = descriptor

        pyplot.clf()
        pyplot.xlabel(transform_axis_label)
        pyplot.ylabel('Distance from original')
        for descriptors, label in [(sift_descriptors, 'SIFT'), (hog_descriptors, 'HOG')]:
            distances = [numpy.linalg.norm(desc - descriptors[0]) for desc in descriptors]
            # Use plot() because of autoscale bug with scatter()
            pyplot.plot(transform_vals, distances, marker='o', ls='', label=label)
        pyplot.legend()
        pyplot.tight_layout()
        output_file = os.path.join(output_dir, DESCRIPTOR_COMPARE_OUTPUT)
        pyplot.savefig(output_file)
