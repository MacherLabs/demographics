from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from demographics.utils import *
#Not needed for mobilenet realted demography
try:
    from demographics.model import select_model, get_checkpoint
    from face import Face
    import common
    import torch
    from torchvision import transforms
    from PIL import Image
    from .coral_cnn import resnet34
    RESIZE_FINAL = 227

    model_type = 'inception'
    checkpoint = 'checkpoint'

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    model_fn = select_model(model_type)

except:
    pass
import cv2

import os
#work_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = "/LFS/demographics"

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

class AgeEstimate_Coral(object):
    def __init__(self):
        RANDOM_SEED = 0
        if torch.cuda.is_available():
            torch.cuda.manual_seed(RANDOM_SEED)
            self.DEVICE = torch.device("cuda:0")
        else:
            torch.manual_seed(RANDOM_SEED)
            self.DEVICE = torch.device("cpu")
        self.model_dir = os.path.join(work_dir, 'model_age_coral')
        self.model = resnet34(100, False)
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir,'model_imdb.pt')))
        self.custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.RandomCrop((120, 120)),
                                       transforms.ToTensor()])
        self.label_list = ['(0, 3)','(4, 7)','(8, 13)','(14, 22)','(23, 34)','(35, 46)','(47, 59)','(60, 100)']
        self.model.to(self.DEVICE)
        self.model.eval()

    def run(self, face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = Image.fromarray(face)
        face = self.custom_transform(face)
        face = face.view(-1, 3, 120, 120)
        if torch.cuda.is_available():
            face = face.type(torch.cuda.FloatTensor)
        else:
            face = face.type(torch.FloatTensor)
        face.to(self.DEVICE)
        logits, pred = self.model(face)
        age = pred > 0.5
        age = torch.sum(age, dim=1)
        best = self.encoded_age(age)
        return self.label_list[best]

    def encoded_age(self, val):
        if(val<=3):
            z = 0
        elif(val<=7):
            z = 1
        elif(val<=13):
            z = 2
        elif(val<=22):
            z = 3
        elif(val<=34):
            z = 4
        elif(val<=46):
            z = 5
        elif(val<=59):
            z = 6
        elif(val<=100):
            z = 7
        return z

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
    
class AgeGenderEstimate_mobilenetv2(object):
    import tensorflow.contrib.tensorrt as trt
    def __init__(self,gpu_frac=0.3):
        
        self.model_dir = os.path.join(work_dir, 'age_gender_frozen_trt_mobilenetv2.pb')
        self.label_list_gender = ['F','M']
        self.label_list_age = ['(0, 3)','(4, 7)','(8, 13)','(14, 22)','(23, 34)','(35, 46)','(47, 59)','(60, 100)']
        self.graph = tf.Graph()
     
        with tf.gfile.GFile(self.model_dir, "rb") as f:
            restored_graph_def = tf.GraphDef()
            restored_graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(restored_graph_def,input_map=None,return_elements=None,name="")
        
        self.gender_logits = graph.get_tensor_by_name('gender/Squeeze:0')
        self.image_placeholder = graph.get_tensor_by_name('Image:0')
        self.age = graph.get_tensor_by_name('Age_final:0')
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
        self.sess= tf.Session(graph=graph,config=config)

    def encoded_age(self, val):
        if(val<=3):
            z = 0
        elif(val<=7):
            z = 1
        elif(val<=13):
            z = 2
        elif(val<=22):
            z = 3
        elif(val<=34):
            z = 4
        elif(val<=46):
            z = 5
        elif(val<=59):
            z = 6
        elif(val<=100):
            z = 7
        return z

    def run(self, face):
        with self.graph.as_default():
           gender_, age_ = self.sess.run([self.gender_logits,self.age],feed_dict={self.image_placeholder:np.expand_dims(face,2)})
           return self.label_list_gender[np.argmax(gender_[0])],self.label_list_age[self.encoded_age(age_[0])]


class Demography(object):
    def __init__(self, face_method='dlib', device='cpu', age_method='mobilenetv2',gender_method='mobilenetv2',gpu_config =0.3):
        if age_method == 'inception':
            self.age_estimator = AgeEstimate()
        elif age_method == 'coral':
            self.age_estimator = AgeEstimate_Coral()
        elif age_method == 'mobilenetv2':
            self.gender_age_estimator = AgeGenderEstimate_mobilenetv2(gpu_frac=gpu_config)

        if gender_method == 'mobilenetv2':
            self.gender_age_estimator = AgeGenderEstimate_mobilenetv2(gpu_frac=gpu_config)
        elif gender_method == 'inception':
            self.gender_estimator = GenderEstimate()
        if face_method:
            self.face_detect = Face(detector_method=face_method, recognition_method=None)

    def run(self, imgcv):
        faces = self.face_detect.detect(imgcv)
        results = []
        for face in faces:
            results.append(self.run_face(imgcv, face['box']))
        return results

    def run_face(self, imgcv, face_box):
        face_image = common.subImage(imgcv, face_box)
        gender = self.gender_estimator.run(face_image)

        if isinstance(self.age_estimator, AgeEstimate):
            age = self.age_estimator.run(face_image)
        elif isinstance(self.age_estimator, AgeEstimate_Coral):
            face_image = common.subImage(imgcv, face_box, padding_type='coral')
            age = self.age_estimator.run(face_image)

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
