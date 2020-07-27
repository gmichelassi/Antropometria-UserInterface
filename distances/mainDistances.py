import copy
import os

import cv2

from distances import calculateDistancesPx
from distances import detectReferenceStripe
from distances import faceNormalizer
import logger
from distances.LandMarkPoints import LandMarkPoints

log = logger.getLogger(__file__)

ROOT_DIR = os.path.abspath(os.getcwd())


def distances_multiplied_by_value(distances, value):
    distances_copy = copy.deepcopy(distances)
    distances_copy.update((x, y * value) for x, y in list(distances_copy.items()))
    return distances_copy


def face_distances(image_path):
    log.info("Image processing started")
    log.info("Switching to dir %s " % ROOT_DIR)
    os.chdir(ROOT_DIR)
    lm = LandMarkPoints()
    log.info('processing %s... ' % image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        log.warning("Failed to load", image_path)
        return None

    try:
        cropped = faceNormalizer.normalize(image_path)
        reference_info = detectReferenceStripe.detect(image_path)

    except:
        log.exception("Failed to process %s", image_path)
        return None

    log.info("Calling external programming for feature extraction")
    if cropped is None:
        img_aux = copy.deepcopy(img)
        points_dict, dir_name = lm.dlib_hog(img)
    else:
        img_aux = copy.deepcopy(cropped)
        points_dict, dir_name = lm.dlib_hog(cropped)

    head, tail = os.path.split(image_path)
    if points_dict is not None:
        for i in range(0, 68):
                x = int(points_dict['x_' + str(i)])
                y = int(points_dict['y_' + str(i)])
                cv2.circle(img_aux, (x, y), 5, (0, 255, 0), -1)
        cv2.imwrite("./output/" + tail, img_aux)
        log.info("Calculating distances px: ALL")
        distances_all_eu_px, distances_all_mh_px = calculateDistancesPx.all(points_dict, reference_info)
        return distances_all_eu_px
    else:
        log.critical("Landmarking extraction did not went well! Finishing image processing...")
        return None


if __name__ == '__main__':
    euclidian = face_distances("MIT-10.jpg")
    print(euclidian)
