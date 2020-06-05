from TrainingandTest import *
from Preprocessing import *


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
save_path = "C:/Users/lenovo system/PycharmProjects/FaceRecognition/CK+48"


def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        images.append(gray)
        print("Added")


if __name__ == '__main__':
    images = []
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cv2.namedWindow('hey')
    cv2.setMouseCallback('hey', mouse_click)

    while True:
        ret, frame = webcam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_ = face_cascade.detectMultiScale(gray)
        for (x, y, w, h) in faces_:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('hey', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    webcam.release()
    histo = []
    name = input()
    img_count = 0
    for img in images:
        img_count += 1
        filename = name + '_' + str(img_count) + '.jpg'
        cv2.imwrite(os.path.join(save_path, filename), img)
        roi_face = preprocess(img)
        if roi_face is None:
            print("One pic down")
            continue
        LBP_image = extended_lbp(roi_face)
        hist = plot_hist(LBP_image)
        histo.append(hist)
        if hist is not None:
            with open('labeldata.csv', 'a') as f:
                f.write(name + '\n')
    f = open('histodata.csv', 'ab')
    np.savetxt(f, histo, fmt='%1.1f', delimiter=',')
