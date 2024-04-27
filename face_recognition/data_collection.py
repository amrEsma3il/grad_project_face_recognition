import cv2
import numpy as np
import os

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0
nameID = str(input("Enter your name: ")).lower()
path = 'images/train/' + nameID
validation_data_dir = 'images/test/' + nameID  # New directory for validation data
isExist = os.path.exists(path)
isValidationExist = os.path.exists(validation_data_dir)

if isExist or isValidationExist:
    print("Name already taken")
    nameID = str(input("Enter your name again: "))
else:
    os.makedirs(path)
    os.makedirs(validation_data_dir)  # Create the validation data directory

while True:
    ret, frame = video.read()

    if not ret:
        print("Failed to capture frame, exiting...")
        break

    faces = facedetect.detectMultiScale(frame, 1.3, 5)
    for x, y, w, h in faces:
        count += 1
        if count <= 500:
            name = './images/train/' + nameID + '/' + str(count) + '.jpg'
        else:
            name = './images/test/' + nameID + '/' + str(count) + '.jpg'
        print("Creating image...", name)
        cv2.imwrite(name, frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow("windowframe", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count > 600:
        break

video.release()
cv2.destroyAllWindows()