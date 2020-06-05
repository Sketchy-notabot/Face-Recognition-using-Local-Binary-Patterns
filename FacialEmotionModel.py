import os
from LBPandHist import *
from Preprocessing import *
from random import shuffle
from sklearn.svm import SVC
import pickle


def convert_file(filename):
    if isinstance(filename, str):
        img = cv2.imread(filename, 0)
    else:
        img = filename
    if img is None:
        return None
    roi_face = preprocess(img)
    if roi_face is None:
        return None
    LBP_image = extended_lbp(roi_face)
    hist = plot_hist(LBP_image)
    return hist


def get_files():
    hist_files = []
    label_files = []
    for foldername in os.listdir(data_path):
        emotion_path = os.path.join(data_path, foldername)
        for filename in os.listdir(emotion_path):
            hist = convert_file(os.path.join(emotion_path, filename))
            if hist is None:
                continue
            hist_files.append(hist)
            label_files.append(foldername)
        print(foldername)
    return hist_files, label_files


if __name__ == '__main__':
    clf = SVC(kernel='linear', probability=True, decision_function_shape='ovo')
    data_path = 'C:/Users/lenovo system/PycharmProjects/FaceRecognition/CK+48'
    histo_files, labe_files = get_files()
#    accuracies = []
    temp = list(zip(histo_files, labe_files))
#    for k in range(5):
    shuffle(temp)
    histo_files, labe_files = zip(*temp)
    X_train = histo_files[:int(len(histo_files)*0.8)]
    Y_train = labe_files[:int(len(labe_files) * 0.8)]
    X_test = histo_files[-int(len(histo_files) * 0.2):]
    Y_test = labe_files[-int(len(labe_files) * 0.2):]
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    clf.fit(X_train, Y_train)
    acc = clf.score(X_test, Y_test)
#    accuracies.append(acc)
#        print(k)
#    print(mean(accuracies))
    print(acc)
    filename = 'emotion_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    print("Done training")


