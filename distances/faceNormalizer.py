import math
import cv2
import numpy as np
from PIL import Image
import cmath
from decimal import *

import logger

log = logger.getLogger(__file__)
SCALE_FACTOR = 1.3
MIN_NEIGHBORS = 5
MIN_SIZE = (20, 20)


def rotate_image(image, angle):
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    rotated_coords = [
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if width > image_size[0]:
        width = image_size[0]

    if height > image_size[1]:
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def eye_coordinate(img, h, w1, w2):
    sum_x = 0
    sum_y = 0
    count_x = 0
    count_y = 0
    for i in range(w1, w2):
        for j in range(0, h):
            px = img[j, i]
            ent = px.astype(np.int)
            if (ent <= 275) and (ent >= 250):
                sum_x = sum_x + i
                sum_y = sum_y + j
                count_x = count_x + 1
                count_y = count_y + 1
    x = sum_x / count_x
    y = sum_y / count_y
    return y, x


def normalize(img_path):
    orig = cv2.imread(img_path)
    colorful = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('./distances/haarcascade_frontalface_alt2.xml')
    log.debug("Loading cascade classifier: %s" % "haarcascade_frontalface_alt2.xml")
    # detect face
    log.debug("Trying to find a face:")
    faces = face_cascade.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBORS, cv2.CASCADE_SCALE_IMAGE, MIN_SIZE)
    log.info("Found {0} faces!".format(len(faces)))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]

    # cut image
    log.info("Cropping detected face:")
    image = Image.open(img_path)
    box = (x, y, x + w, y + h)

    face_box = image.crop(box)
    face = np.asarray(face_box)
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # detect eyes pair
    log.info("Detect eyes pair:")

    eye_cascade = cv2.CascadeClassifier('./distances/haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(face_gray)
    log.debug("Trying to find eyes pair:")
    log.info("Found {0} eyes pair!".format(len(eyes)))
    eyes_pair = None
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(face_gray, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        eyes_pair = face_gray[ey:ey + eh, ex:ex + ew]
        eyes_box = (ex, ey, ex + ew, ey + eh)
    if eyes_pair is None:
        return None

    ################################# detect eyes borders
    log.debug("Detecting eyes borders")
    canny = cv2.Canny(eyes_pair, 50, 245)
    kernel = np.ones((3, 3), np.uint8)
    gradient = cv2.morphologyEx(canny, cv2.MORPH_GRADIENT, kernel)

    ################################# find coordinates of the pupils
    h = gradient.shape[0]
    w = gradient.shape[1]
    ################################# normalize image geometrically
    y1, x1 = eye_coordinate(gradient, h, 0, int(w / 2))  # left eye
    log.debug("Normalizing geometrically:")
    y1 = abs(y1)
    x1 = abs(x1)
    y2, x2 = eye_coordinate(gradient, h, int(w / 2), w)  # right eye
    y2 = abs(y2)
    x2 = abs(x2)
    dy = y2 - y1
    dx = x2 - x1
    dy = abs(dy)
    dx = abs(dx)
    dy = dy * (-1)
    z = Decimal(dy) / Decimal(dx)
    alpha_complex = cmath.atan(z)
    alpha = cmath.phase(alpha_complex)
    alpha = alpha / 2
    log.debug("Alpha angle to rotate image: %f" % alpha)
    log.info("Rotating and saving output image:")
    img = Image.open(img_path)
    rotated_image = img.rotate(-alpha)
    rotated_image = cv2.cvtColor(np.array(rotated_image), cv2.COLOR_RGB2BGR)
    log.info("Cropping and saving final normalized image")
    image_height, image_width = orig.shape[0:2]

    image_rotated_cropped = crop_around_center(
        rotated_image,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(-alpha)
        )
    )

    return image_rotated_cropped


if __name__ == '__main__':
    print("Image pre-processor module")
