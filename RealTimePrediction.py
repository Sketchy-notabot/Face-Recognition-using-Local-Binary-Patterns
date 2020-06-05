from FacialEmotionModel import *
from TrainingandTest import *
from Classification import *

#train_model()
# test_model(2)

dict = {0: 'Angry', 1: 'Disgusted', 2: 'Happy', 4: 'Sad', 3: 'Surprised', 5: 'Neutral'}
filename = 'emotion_model.sav'
clf = pickle.load(open(filename, 'rb'))


#click to predict name
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img = param[0]
        try:
            name = predict(img)
            print(name)
        except ValueError:
            print("Could not predict")


webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow('hey')
cv2.setMouseCallback('hey', mouse_click)
frame_count = 1
k = 5
hash = [0, 0, 0, 0, 0, 0]
i = 5
while True:
    ret, frame = webcam.read()
    frame_count += 1
    cv2.putText(frame, str(frame_count), (19, 19), cv2.FONT_HERSHEY_SIMPLEX, 1, (156, 127, 134))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_ = face_cascade.detectMultiScale(gray)
    histo = []
    for (x, y, w, h) in faces_:
        cv2.setMouseCallback('hey', mouse_click, param=gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        hist = convert_file(gray)
        if hist is None:
            continue
        histo.append(hist)
        listt = clf.predict_proba(histo)[0]
        prob = max(listt)
        if prob > 0.25:
            i = int(np.argmax(listt))
        else:
            i = 5
    cv2.putText(frame, dict[i], (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 123, 0))
    cv2.imshow('hey', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
webcam.release()
