import cv2 as cv
import numpy

def adjust_for_scale(image_size, scale, point):
    T = [[1 / scale, 0],
         [0, 1 / scale]]
    return numpy.dot(point, T)

def adjust_for_rotation(image_size, angle, point):
    centre = (image_size[0] / 2, image_size[1] / 2)
    T = cv.getRotationMatrix2D(centre, -angle, 1)
    return numpy.dot(T, [point[0], point[1], 1])

BASE_IMAGES = ['card', 'dugong']
INPUTS = [
    ('scaled', 'results/{}-scaled_{}x.png', [1, 1.25, 1.5, 2], adjust_for_scale),
    ('rotated', 'results/{}-rotated_{}deg.png', [0, 15, 45, 75], adjust_for_rotation)
]
KEYPOINT_MATCH_OUTPUT = 'results/{}-{}-keypoint_matches.png'
KEYPOINT_MATCH_THRESHOLD = 1

for transform_name, file_template, transforms, keypoint_adjuster in INPUTS:
    for base_image in BASE_IMAGES:
        # METHODOLOGY:
        # To choose a keypoint to examine, we first get all keypoints from all versions of the image.
        # We then apply to the keypoints the inverse of the transformation that was applied to that image, to get the
        # locations where the keypoints "should be" if SIFT was perfectly scale/rotation invariant.
        # Finally, we select keypoints that match in location across all versions of the image.

        # TODO: merge actual and adjusted keypoints so they are associated
        actual_keypoints = {}
        adjusted_keypoints = {}
        for transform in reversed(transforms):
            file = file_template.format(base_image, transform)
            image = cv.imread(file, cv.IMREAD_COLOR)
            sift = cv.xfeatures2d.SIFT_create()
            keypoints = sift.detect(image, None)
            actual_keypoints[transform] = keypoints
            # Find where the keypoints should be in the original image.
            adjusted_keypoints[transform] = [keypoint_adjuster((image.shape[1], image.shape[0]), transform, kp.pt)
                                             for kp in keypoints]

        # Find keypoints that match in location across all the image versions.
        matches = actual_keypoints[transforms[0]]
        for s in transforms[1:]:
            matches = [kp for kp in matches for pt in adjusted_keypoints[s]
                       if numpy.linalg.norm(kp.pt - pt) <= KEYPOINT_MATCH_THRESHOLD]

        result = cv.drawKeypoints(image, matches, None)
        output_file = KEYPOINT_MATCH_OUTPUT.format(base_image, transform_name)
        cv.imwrite(output_file, result)

        # Choose strongest keypoint to examine.
        target_keypoint = max(matches, key=lambda kp: kp.size)

        # TODO: compare keypoint across transformations
