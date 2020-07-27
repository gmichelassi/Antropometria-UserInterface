import cv2
import dlib
from imutils import face_utils


class LandMarkPoints:
    dlib_land_marks_extractor = "./distances/shape_predictor_68_face_landmarks.dat"
    dlib_cnn_file = "./distances/mmod_human_face_detector.dat"

    def dlib_hog(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hogFaceDetector = dlib.get_frontal_face_detector()
        faceRects = hogFaceDetector(img, 0)
        points = None
        for faceRect in faceRects:
            x1 = faceRect.left()
            y1 = faceRect.top()
            x2 = faceRect.right()
            y2 = faceRect.bottom()
            w = x2 - x1
            h = y2 - y1
            face = x1, y1, w, h
            points = self.extract_land_marks(face, gray)
        return points, '/dlibCNN/'

    def extract_land_marks(self, face, gray):
        predictor = dlib.shape_predictor(self.dlib_land_marks_extractor)
        x, y, w, h = face
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, dlib_rect)
        shape = face_utils.shape_to_np(shape)
        points = {}
        for (i, (x, y)) in enumerate(shape):
            points['x_' + str(i)] = x
            points['y_' + str(i)] = y
        return points
