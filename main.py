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

def blob_process(img, detector, threshold):
    """ Detect the blob of the eye """
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2) #1
    img = cv2.dilate(img, None, iterations=4) #2
    img = cv2.medianBlur(img, 5) #3
    keypoints = detector.detect(img)
    return keypoints


def main():
    # First need the classifiers
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
    # Blob detector
    detector_params = cv2.SimpleBlobDetector_Params()
    detector_params.filterByArea = True
    detector_params.maxArea = 1500
    detector_params.filterByConvexity = False
    detector_params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(detector_params)
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('jeu')
    cv2.createTrackbar('threshold', 'jeu', 0, 255, lambda x: 0)
    
    while True:
        ret, frame = cap.read()
        face_frame = detect_face(frame, face_cascade)
        if face_frame is not None:
            eyes = detect_eyes(face_frame, eye_cascade)
            for eye in eyes:
                if eye is not None:
                    threshold = cv2.getTrackbarPos('threshold', 'jeu')
                    eye = cut_eyebrows(eye)
                    keypoints = blob_process(eye, detector, threshold)
                    eye= cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('jeu', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
if __name__=="__main__":
    main()
