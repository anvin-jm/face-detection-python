import cv2
from random import randrange

# loading pre-saved data (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv2.imread('ajz.jpg')
# if you give it an file name instead of 0 as an argument it will capture the video
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# videos are images combined together , so we are iterating it.
while True:
    # this here returns 2 values a boolean(to show if the frame was successful or not) and the real frame
    frame_success, frame = cam.read()

    # converting image to grayscale
    gs_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # face detection
    face_coordinates = trained_face_data.detectMultiScale(gs_img)
    # print(face_coordinates)

    # creating a rectangle around the face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 3)

    cv2.imshow('face detection sample', frame)  # shows the image
    key = cv2.waitKey(1)  # waits till a key is pressed to stop showing
    # here the wait key presses a key every 1 milli second

    if key == 81 or key == 113:
        break

cam.release()

print("completed")
