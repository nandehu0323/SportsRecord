import numpy as np
import os
import sys
import argparse
import glob
import time

import caffe

def predict(file):
    model_def = "./data/mnist/lenet.prototxt"
    pretrained_model = "./data/mnist/lenet_iter_10000.caffemodel"

    caffe.set_mode_gpu()

    # Make classifier.
    classifier = caffe.Classifier(model_def, pretrained_model)

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    #grayimg = caffe.io.load_image(file, color=False)[:,:,0]
    inputs = [np.reshape(file, (28, 28, 1))]

    # Classify.
    start = time.time()
    predictions = classifier.predict(inputs)
    print(predictions)
