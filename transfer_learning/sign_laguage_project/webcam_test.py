import time

import cv2

from keras.applications import vgg16
from keras.layers import Dense
from keras.models import Model

import threading

import numpy as np


# load the NN model
base_model = vgg16.VGG16(weights = "imagenet", include_top=False, input_shape = (224,224, 3), pooling='avg')
last_layer = base_model.get_layer('global_average_pooling2d')
last_output = last_layer.output
x = Dense(10, activation='softmax', name='softmax')(last_output)
new_model = Model(inputs=base_model.input, outputs=x)
new_model.load_weights('./sign_laguage_project/signlanguage.model.hdf5')
new_model.summary()


# create independent thread with a queue for processing frames
class FrameProcessor(threading.Thread):
    def __init__(self, model):
        threading.Thread.__init__(self)
        self.model = model
        self.frames_queue = []
        self.isProcessFrames = True
        self.actual_frame_data = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def add_frame(self, new_frame):
        self.frames_queue.append(new_frame)
        if len(self.frames_queue) > 2:
            self.frames_queue.pop(0)

    def stop(self):
        self.isProcessFrames = False

    def run(self):
        self.isProcessFrames = True
        while self.isProcessFrames:
            if len(self.frames_queue) == 0:
                time.sleep(0.01)
            else:
                p_frame = self.frames_queue.pop(0)
                p_frame = cv2.resize(p_frame, (224, 224))
                x_data = np.array([p_frame])
                model_prediction = self.model.predict(x_data)
                model_prediction = np.array(model_prediction[0])
                self.actual_frame_data = model_prediction * 100


frame_processor = FrameProcessor(model=new_model)
frame_processor.start()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame_processor.add_frame(frame)

    prediction = frame_processor.actual_frame_data
    selected = np.argmax(prediction)

    for i in range(len(prediction)):
        color = (0, 255, 0) if i == selected else (0, 0, 255)
        cv2.putText(img=frame, text=(str(i) + ' ~> ' + str(prediction[i])), org=(10, 15 + i * 18), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6, thickness=1, color=color)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        frame_processor.stop()
        break


cap.release()
cv2.destroyAllWindows()