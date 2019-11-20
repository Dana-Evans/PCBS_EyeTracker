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
            # cv2.rectangle(img,(x, y), (x+w, y+h), EYES_COLOR[0], 2)
        else:
            right_eye = img[y:y + h, x:x + w]
            # cv2.rectangle(img,(x, y), (x+w, y+h), EYES_COLOR[1], 2)
        
    return left_eye, right_eye

def detect_face(img, classifier):
    """ Function that detect the biggest face on the image and returns it """
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = classifier.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) == 0:
        return None # No faces detected
    biggest = (0, 0, 0, 0)
    for i in coords:
        if i[3] > biggest[3]:
            biggest = i # Largest width is chosen (Could have been width and height combined also)
    x,y,w,h = biggest
    return img[y:y + h, x:x + w]

def cut_eyebrows(img):
    """ Remove the eyebrows from the eye """
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    return img[eyebrow_h:height, 0:width]

if __name__=="__main__":
    # First need the classifiers
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    # Load the image
    img = cv2.imread("test2.jpg")

    # Transformations of the image
    gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # put the picture in gray scale
    retval, img_black_and_white = cv2.threshold(gray_picture, 140, 255, cv2.THRESH_BINARY)

    # Detection of face and eyes
    face = detect_face(img, face_cascade)
    left_eye, right_eye = detect_eyes(face, eye_cascade)

    left_eye_without_eyebrow = cut_eyebrows(left_eye)
    right_eye_without_eyebrow = cut_eyebrows(right_eye)

    # Blob detection
    detector_params = cv2.SimpleBlobDetector_Params()
    detector_params.filterByArea = True
    detector_params.maxArea = 1500

    detector = cv2.SimpleBlobDetector_create(detector_params)

    # Showing resultss
    cv2.imshow('pft', left_eye_without_eyebrow)
    cv2.waitKey(0) # let the window open until a key is pressed

    cv2.imshow('pft', face)
    cv2.waitKey(0) # let the window open until a key is pressed

    cv2.imshow('pft', left_eye)
    cv2.waitKey(0) # let the window open until a key is pressed

    cv2.imshow('pft', right_eye)
    cv2.waitKey(0) # let the window open until a key is pressed

    cv2.destroyAllWindows() 
