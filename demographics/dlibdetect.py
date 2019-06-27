import dlib
import cv2
FACE_PAD = 50

class FaceDetectorDlib(object):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        # self.predictor = dlib.shape_predictor(model_name)

    def run(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        result = []
        for (i, rect) in enumerate(faces):
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y
            face = dict()
            face['rect'] = ((x,y),(w+x,h+y))
            face['image'] = self.sub_image(img, x, y, w, h)
            result.append(face)

        # print('%d faces detected' % len(result))
        return result

    def sub_image(self, img, x, y, w, h):
        upper_cut = [min(img.shape[0], y + h + FACE_PAD), min(img.shape[1], x + w + FACE_PAD)]
        lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
        roi_color = img[lower_cut[0]:upper_cut[0], lower_cut[1]:upper_cut[1]]
        return roi_color