import colorsys
import os
import sys
sys.path.append("..")

import shutil
from timeit import default_timer as timer
import pdb
p=pdb.set_trace
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import cv2

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.utils import multi_gpu_model
gpu_num=1

class YOLO(object):
    def __init__(self):
        self.model_path = '../logs/002/ep036-loss29.343-val_loss29.638.h5' # model path or trained weights path
        self.anchors_path = '../model_data/yolo_anchors.txt'
        self.classes_path = '../model_data/coco_class.txt'
        self.score = 0.15
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()
        ret = []
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        #print(out_boxes, out_scores, out_classes)
        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
 
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #print(label, (left, top), (right, bottom))
            
            ret.append([predicted_class, score, left, top, right, bottom])

        return ret
    
    
    def detect_image2(self, image):
        start = timer()
        ret = []
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        #print(out_boxes, out_scores, out_classes)
        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
 
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #print(label, (left, top), (right, bottom))
            
            ret.append(f"{predicted_class} {score} {left} {top} {right} {bottom}")

        return ret

    def close_session(self):
        self.sess.close()

def detect_img(yolo):
    #img = input('Input image filename:')
    img = "./test" + os.sep + "11_z_2018_12_13_11_18_04_357939_44.jpg"
    try:
        image = Image.open(img)
#         image1 = cv2.imread(img)
        
#         image = cv2.imread(img)
#         image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        print('Open Error! Try again!')
    else:
        r_image = yolo.detect_image(image)
        print(r_image)
        #r_image.show()
        
        for aa in r_image:
            if aa[1] < 0.5: # 设定得分低于0.5的 删除
                r_image.remove(aa)
        if len(r_image) != 5:
            print(False)
        else:
            print(True)
#             cv2.rectangle(image1, (aa[2], aa[3]), (aa[4], aa[5]),(255,255,0), 3)
#             cv2.putText(image1, aa[0], (aa[2], aa[3]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 2)
            
#         cv2.namedWindow("aaa", cv2.WINDOW_NORMAL)
#         cv2.imshow("aaa", image1)
#         cv2.waitKey(0)
            
    yolo.close_session()


def mAP(yolo,anno_dir):
    '''读取目录图片，图片需要跟truth的完全一致，调用自己训练的模型计算
    '''
    predir = "./detection-results"
    if os.path.exists(predir):
        shutil.rmtree(predir)
    os.mkdir(predir)
    import cv2
    with open(anno_dir) as lines:
        for line in lines:
            tmp_line=line.split(' ')
            imagefile=tmp_line[0]
            file=tmp_line[0].split('/')[-1]
            #image = Image.open(imagefile)
            image = cv2.imread(imagefile)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            r_image = yolo.detect_image2(image)
            print(r_image)
            #p()
            prefile = open(predir + os.sep + file.split(".")[0]+".txt", "w")
            [prefile.write(i+"\n") for i in r_image]
            prefile.close()
        
        
    yolo.close_session()

if __name__ == '__main__':
    mAP(YOLO(),"../low_test.txt")