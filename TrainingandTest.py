import cv2
import numpy as np
import os
from LBPandHist import *
from Classification import *
from Preprocessing import *


training_path = "C:/Users/lenovo system/PycharmProjects/FaceRecognition/yalefaces/Training"
testing_path = "C:/Users/lenovo system/PycharmProjects/FaceRecognition/yalefaces/Testing"


def train_model():
    images = []
    labels = []
    histograms = []
    for filename in os.listdir(training_path):
        if filename.split('.')[-1] == 'jpg':
            img = cv2.imread(os.path.join(training_path, filename), 0)
        else:
            gif = imageio.mimread(os.path.join(training_path, filename))
            img = gif[0]
        roi_face = preprocess(img)
        images.append(roi_face)
        LBP_image = extended_lbp(roi_face)
#       LBP_image = only_uniform(LBP_image)
        hist = plot_hist(LBP_image)
        histograms.append(hist)
        labels.append(filename.split('.')[0])
        print(filename.split('.')[0] + " updated")
    np.savetxt('histodata.csv', histograms, delimiter=',', fmt='%1.1f')
    np.savetxt('labeldata.csv', labels, delimiter=',', fmt="%s")
    print("model trained")


def test_model(metric=2):
    correct = 0
    total = 0
    for filename in os.listdir(testing_path):
        if filename.split('.')[-1] == 'jpg':
            img = cv2.imread(os.path.join(testing_path, filename), 0)
        else:
            gif = imageio.mimread(os.path.join(testing_path, filename))
            img = gif[0]
        pred_name = predict(img, metric)
        if pred_name is None:
            print(filename)
        img_name = filename.split('.')[0]
        if pred_name == img_name:
            correct += 1
        total += 1
    acc = correct / total
    acc = acc*100
    print("Accuracy: " + str(acc) + "%")


if __name__ == '__main__':
    pass


