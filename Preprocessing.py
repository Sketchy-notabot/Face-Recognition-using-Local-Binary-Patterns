import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
dimensions = (250, 250)


def hitogramequaization(img):
    h, b = np.histogram(img.flatten(), 256)
    cs = h.cumsum()
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()
    cs = nj / N
    cs = cs.astype('uint8')
    imgo = cs[ img.flatten() ]
    imgo = np.reshape(imgo, img.shape)
    return imgo


def preprocess(img):
    faces_ = face_cascade.detectMultiScale(img)
    if len(faces_) == 0:
        print("No face detected")
        return None
    face_count = 0
    for (x, y, w, h) in faces_:
        if face_count > 0:
            break
    img = img[y:y + h, x:x + w]
    face_count += 1
    imgo = hitogramequaization(img)
    imgo = cv2.resize(imgo, dimensions)
    return imgo


if __name__ == '__main__':
    pass