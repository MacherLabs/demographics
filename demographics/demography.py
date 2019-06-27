from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from demographics.utils import *
from demographics.model import select_model, get_checkpoint
from face import Face
import common

import os
#work_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = "/LFS"

RESIZE_FINAL = 227

model_type = 'inception'
checkpoint = 'checkpoint'

gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
model_fn = select_model(model_type)


def classify(sess, label_list, softmax_output, images, image_batch, face_placeholder, face):
    try:
        batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval(session=sess, feed_dict={face_placeholder:face})})
        output = batch_results[0]
        batch_sz = batch_results.shape[0]

        for i in range(1, batch_sz):
            output = output + batch_results[i]

        output /= batch_sz
        best = np.argmax(output)
        best_choice = (label_list[best], output[best])
        return label_list[best]

    except Exception as e:
        print(e)
        print('Failed to run image')
        return None


class AgeEstimate(object):
    def __init__(self):
        self.model_dir = os.path.join(work_dir, 'model_age')
        self.label_list = ['(0, 3)','(4, 7)','(8, 13)','(14, 22)','(23, 34)','(35, 46)','(47, 59)','(60, 100)']
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
            logits = model_fn(len(self.label_list), self.images, 1, False)
            init = tf.global_variables_initializer()
            model_checkpoint_path, global_step = get_checkpoint(self.model_dir, None, checkpoint)
            self.softmax_output = tf.nn.softmax(logits)

            self.sess = tf.Session(config=config,graph=self.graph)
            saver = tf.train.Saver()
            saver.restore(self.sess, model_checkpoint_path)

            self.face_placeholder = tf.placeholder("uint8", [None, None, 3])
            self.image_batch = make_multi_crop_batch(self.face_placeholder)

    def run(self, face):
        with self.graph.as_default():
            return classify(self.sess, self.label_list, self.softmax_output, self.images, self.image_batch, self.face_placeholder, face)


class GenderEstimate(object):
    def __init__(self):
        self.model_dir = os.path.join(work_dir, 'model_gender')
        self.label_list = ['M','F']
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
            logits = model_fn(len(self.label_list), self.images, 1, False)
            init = tf.global_variables_initializer()
            model_checkpoint_path, global_step = get_checkpoint(self.model_dir, None, checkpoint)
            self.softmax_output = tf.nn.softmax(logits)

            self.sess = tf.Session(config=config,graph=self.graph)
            saver = tf.train.Saver()
            saver.restore(self.sess, model_checkpoint_path)

            self.face_placeholder = tf.placeholder("uint8", [None, None, 3])
            self.image_batch = make_multi_crop_batch(self.face_placeholder)

    def run(self, face):
        with self.graph.as_default():
            return classify(self.sess, self.label_list, self.softmax_output, self.images, self.image_batch, self.face_placeholder, face)


class Demography(object):
    def __init__(self, face_method='dlib', device='cpu'):
        self.age_estimator = AgeEstimate()
        self.gender_estimator = GenderEstimate()
        self.face_detect = Face(detector_method=face_method, recognition_method=None)

    def run(self, imgcv):
        faces = self.face_detect.detect(imgcv)
        results = []
        for face in faces:
            results.append(self.run_face(imgcv, face['box']))
        return results

    def run_face(self, imgcv, face_box):
        face_image = common.subImage(imgcv, face_box)
        age = self.age_estimator.run(face_image)
        gender = self.gender_estimator.run(face_image)
        return self._format_results(face_box, age, gender)

    def _format_results(self, face_box, age, gender):
        out = {
                'box': face_box,
                'classes': [{'name': 'face',
                             'prob': None,
                             'meta': {'age': age,
                                      'gender': gender
                                      }
                            }]
            }
        return out


def demo_video(video_file):
    import cv2
    # with tf.device(device_id):
    cap = common.VideoStream(video_file).start()
    dmg = Demography()

    def draw_faces(img, detections):
        """ Draws bounding boxes of objects detected on given image """
        h, w = img.shape[:2]
        for face in detections:
            # draw rectangle
            x1, y1, x2, y2 = face['box']['topleft']['x'], face['box']['topleft']['y'], face['box']['bottomright']['x'], face['box']['bottomright']['y']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

            # draw class text
            text = "%s %s"%(face['classes'][0]['meta']['gender'], face['classes'][0]['meta']['age'])
            return common.draw_label(img, text, (x1, y1))

    while cap.more():
        img = cap.read()
        if img is not None:
            faces = dmg.run(img)
            img = draw_faces(img, faces)

            common.showImage(img)
            key = cv2.waitKey(1) & 0xff
            if key == 27:
                break
        else:
            print('Cannot read frame')
            break


if __name__ == '__main__':
    import cv2
    filename = '/home/aestaq/Pictures/general object detection/samples/peds-001.jpg'
    dmg = Demography()
    img = cv2.imread(filename)
    if img is not None:
        print(dmg.run(img))

    # video_file = 0 if len(sys.argv) < 2 else sys.argv[1]
    # video_file = '/home/aestaq/Videos/qb.mp4'
    # demo_video(video_file)
