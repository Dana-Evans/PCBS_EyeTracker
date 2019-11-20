import cv2
import numpy as np

HEAD_COLOR  = (255,255,0)
EYES_COLOR = ((0,255,255), (255,0,255))

def detect_eyes(img, classifier):
    """ Function that detect the eyes, so only in the first half of the face and return the right vs left eye """
    gray_frames = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = classifier.detectMultiScale(gray_frames, 1.3, 5)
    height = np.size(img, 0) # get the width and height of the face image
    width  = np.size(img, 1)
    left_eye, right_eye = None, None # In case no eyes are detected...
    for(x, y, w, h) in eyes :
        if y > height/2: # pass if the eye is not on the top of the face
            pass
        
        eyecenter = x + w /2 # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
            cv2.rectangle(img,(x, y), (x+w, y+h), EYES_COLOR[0], 2)
        else:
            right_eye = img[y:y + h, x:x + w]
            cv2.rectangle(img,(x, y), (x+w, y+h), EYES_COLOR[1], 2)
        
    return left_eye, right_eye

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread("test2.jpg")

gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # put the picture in gray scale
faces = face_cascade.detectMultiScale(gray_picture, 1.3, 5) # coordinates and size of the face(s) [[x,y,w,h],]

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y), (x+w, y+h), HEAD_COLOR, 2) # draw a rectangle on the image for each face detected
    left_eye, right_eye = detect_eyes(img, eye_cascade)
    
cv2.imshow('test', img) #display the result
cv2.waitKey(0) # let the window open until a key is pressed
cv2.destroyAllWindows() 
