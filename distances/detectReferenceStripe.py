import cv2
import imutils
import numpy as np
import os
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist
import logger

log = logger.getLogger(__file__)


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def detect(image_path):
    image = cv2.imread(image_path)
    print("image not cropped")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    log.debug("Applying Gaussian Blur")
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    log.debug("Detecting edges with Canny")
    v = np.median(image)
    sigma = 0.85
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(blurred, lower, v)
    log.debug("Performing dilation and erosion...")
    edged = cv2.dilate(edged, None, iterations=1)
    # edged = cv2.erode(edged, None, iterations=1)
    # find contours in the edge map
    log.debug("Finding countours")
    h, w, channels = image.shape
    y = 0
    x = 0
    h = int(h / 5)
    crop = edged.copy()[y:y + h, x:x + w]
    cnts = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    log.debug("Sorting countours: top-to-bottom")
    # print(cnts)
    # cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * image.shape[1] )
    (cnts, _) = contours.sort_contours(cnts, "top-to-bottom")
    pixelsPerMetric = None
    # I will only use the most top contourArea that is the reference stripe
    c = cnts[0]
    i = 1

    while cv2.contourArea(c) < 300 and i < len(cnts):
        # if the contour is not sufficiently large, ignore it
        c = cnts[i]
        i = i + 1

    if cv2.contourArea(c) < 300:
        edged = cv2.erode(edged, None, iterations=1)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = cnts[0]
        i = 1
        while cv2.contourArea(c) < 300 and i < len(cnts):
            # if the contour is not sufficiently large, ignore it
            c = cnts[i]
            i = i + 1

    log.debug("c area: " + str(cv2.contourArea(c)))
    # compute the rotated bounding box of the contour
    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 1)

    # loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 2, (0, 0, 255), -1)

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    log.debug("Calculating coordinates")
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and bottom-left points,
    # followed by the midpoint between the top-right and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 2, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 2, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 2, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 2, (255, 0, 0), -1)

    # draw lines between the midpoints
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 0, 255), 30)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 0, 255), 30)

    log.debug("Computing euclidean distance between points in the reference stripe...")
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, centimeters)
    if pixelsPerMetric is None:
        pixelsPerMetric = (dB / 50.0)  # in case something go wrong
        pixelsPerMetric2 = (dA / 5.5)

    # compute the size of the object
    dimA = dA / pixelsPerMetric2
    dimB = dB / pixelsPerMetric

    log.info("Reference stripe width in pixels (%.2f) and centimeters (%.2f)" % (dB, dimB))
    log.info("Pixels per metric: (%.2f)" % (pixelsPerMetric))
    return_dict = {"w-pixels": dB, "w-centimeters": dimB, "h-pixels": dA, "h-centimeters": dimA,
                   "pixelsPerMetric": pixelsPerMetric, "coordinates": [tl, tr, br, bl]}
    log.info("Return dict: " + str(return_dict))
    return return_dict


if __name__ == '__main__':
    print("Detect a reference stripe in top of the image module")
