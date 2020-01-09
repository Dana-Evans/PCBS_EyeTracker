"""
The main program that is used to play the game.

:author: `Dana Ladon <dana.ladon@ens.fr>`_

:date:  2019, december

"""
import random

import cv2
import numpy as np
import pygame

HEAD_COLOR = (255, 255, 0)
EYES_COLOR = ((0, 255, 255), (255, 0, 255))
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (127, 127, 127)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def detect_eyes(img, classifier):
    """
    Function that detect the eyes, so only in the first half of the face and return the right vs left eye.

    :param img: The image where the eyes should be located
    :type img: numpy.ndarray
    :param classifier: The classifier used to detect the eyes
    :type classifier: cv2.CascadeClassifier
    :returns: The right eye and the left eye
    :rtype: numpy.ndarray|None

    :UC: None

    .. testsetup::

        from main import *
        import cv2
        from numpy import ndarray

    .. doctest::

        >>> img = cv2.imread("data/test.jpeg")
        >>> face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        >>> eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
        >>> face = detect_face(img, face_cascade)
        >>> left_eye, right_eye = detect_eyes(face, eye_cascade)
        >>> # In this case it is true but might be None also
        >>> isinstance(left_eye, ndarray) and isinstance(right_eye, ndarray)
        True
    """
    gray_frames = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = classifier.detectMultiScale(gray_frames, 1.3, 5)
    height = np.size(img, 0)  # get the width and height of the face image
    width = np.size(img, 1)
    left_eye, right_eye = None, None  # In case no eyes are detected...
    for (x, y, w, h) in eyes:
        if y > height / 2:  # pass if the eye is not on the top of the face
            pass

        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]

    return left_eye, right_eye


def detect_face(img, classifier):
    """
    Function that detect the biggest face on the image and returns it.

    :param img: The image where the face should be located
    :type img: numpy.ndarray
    :param classifier: The classifier used to detect the face
    :type classifier: cv2.CascadeClassifier
    :returns: The image cropped around the face
    :rtype: numpy.ndarray|None

    :UC: None

    .. doctest::

        >>> img = cv2.imread("data/test.jpeg")
        >>> face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        >>> face = detect_face(img, face_cascade)
        >>> isinstance(face, ndarray)
        True
    """
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = classifier.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) == 0:
        return None  # No faces detected
    biggest = (0, 0, 0, 0)
    for i in coords:
        if i[3] > biggest[3]:
            biggest = i  # Largest width is chosen (Could have been width and height combined also)
    x, y, w, h = biggest
    return img[y:y + h, x:x + w]


def cut_eyebrows(img):
    """
    Remove the eyebrows from the eye.

    :param img: One eye where you want to remove the eyebrow from the image
    :type img: numpy.ndarray
    :returns: The image without the eyebrow
    :rtype: numpy.ndarray

    .. doctest::

        >>> img = cv2.imread("data/test.jpeg")
        >>> res = cut_eyebrows(img)
        >>> height, width = img.shape[:2]
        >>> res_height, res_width = res.shape[:2]
        >>> res_height == 3*height // 4
        True
        >>> width == res_width
        True
        >>> isinstance(img, ndarray)
        True
        >>> isinstance(res, ndarray)
        True
    """
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    return img[eyebrow_h:height, 0:width]


def blob_process(img, detector, threshold):
    """
    Detect the blob of the eye.

    :param img: The eye image without eyebrows
    :type img: numpy.ndarray
    :param detector: The detector object used to find the blob
    :type detector: cv2.SimpleBlobDetector
    :param threshold: The threshold we use to darken or lighten the black and white image
                      (a threshold of 10, would convert any pixel below 10 to  0 and above 10 to 255)
    :type threshold: int
    :returns: A list of keypoints or None if not detected
    :rtype: list|None

    :UC: 0 <= threshold <= 255

    .. doctest::

        >>> detector_params = cv2.SimpleBlobDetector_Params()
        >>> detector_params.filterByArea = True
        >>> detector_params.maxArea = 1500
        >>> detector_params.filterByConvexity = False
        >>> detector_params.filterByInertia = False
        >>> detector = cv2.SimpleBlobDetector_create(detector_params)
        >>> img = cv2.imread("data/test.jpeg")
        >>> face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        >>> eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
        >>> face = detect_face(img, face_cascade)
        >>> left_eye, right_eye = detect_eyes(face, eye_cascade)
        >>> left_eye = cut_eyebrows(left_eye)
        >>> keypoints = blob_process(left_eye, detector, 42)
        >>> isinstance(keypoints, list)
        True
        >>> isinstance(keypoints[0], cv2.KeyPoint)
        True
    """
    assert 0 <= threshold <= 255, f"Threshold is too high({threshold}). It should be between 0 and 255 "
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)  # 1
    img = cv2.dilate(img, None, iterations=4)  # 2
    img = cv2.medianBlur(img, 5)  # 3
    keypoints = detector.detect(img)
    return keypoints


def draw_cross(screen, centerx, centery, width, height, crossthickness=5, color=BLUE):
    """
    A simple function to draw a cross on the screen

    :param screen: The screen where we want to do the cross
    :type screen: pygame.Surface
    :param centerx: The x position of the center of the cross
    :type centerx: int
    :param centery: The y position of the center of the cross
    :type centery: int
    :param width: the width of the cross
    :type width: int
    :param height: the height of the cross
    :type height: int
    :param crossthickness: the thickness of the lines of the cross (which are here rectangles in fact)
    :type crossthickness: int
    :param color: A color tuple that contains 3 values between 0 and 255
    :type color: tuple

    :UC: None

    """
    pygame.draw.rect(screen, color,
                     (centerx - crossthickness // 2, centery - height // 2, crossthickness, height))

    pygame.draw.rect(screen, color,
                     (centerx - width // 2, centery - crossthickness // 2, width, crossthickness))


def setup_crosses(step, screen, W, H):
    if step == 0:
        # Middle cross
        draw_cross(screen, W // 2, H // 2, 50, 50)
    elif step == 1:
        # Top cross
        draw_cross(screen, W // 2, H // 8, 50, 50)
    elif step == 2:
        # Right cross
        draw_cross(screen, 7 * W // 8, H // 2, 50, 50)
    elif step == 3:
        # Bottom cross
        draw_cross(screen, W // 2, 7 * H // 8, 50, 50)
    elif step == 4:
        # Left cross
        draw_cross(screen, W // 8, H // 2, 50, 50)


def setup_detector():
    detector_params = cv2.SimpleBlobDetector_Params()
    detector_params.filterByArea = True
    detector_params.maxArea = 1500
    detector_params.filterByConvexity = False
    detector_params.filterByInertia = False
    return cv2.SimpleBlobDetector_create(detector_params)


def get_new_radius(initial_radius: int, radius: int, thresholds: dict, position: [tuple, tuple],
                   center_position: [tuple, tuple]) -> int:
    if not all(position):
        return radius  # Could not capture eyes position

    dx_right = position[1][0] - center_position[1][0]
    dx_left = position[0][0] - center_position[0][0]
    # dy_right = position[1][1] - center_position[1][1]
    # dy_left = position[0][1] - center_position[0][1]

    # Cannot take into account top and bottom because it is too sensitive
    if dx_right < thresholds['right'] or \
            dx_left > thresholds['left']:
        return initial_radius

    if radius < 5:
        print("Woah user won !")
        exit(0)
    else:
        return radius - 5


def get_values_from_eye(eye, blob_process, detector, threshold, current_eye_position, capture_position, keys, step,
                        eyes_position, eye_name):
    p = 1 if eye_name == "right" else 0

    eye = cut_eyebrows(eye)
    keypoints = blob_process(eye, detector, threshold)
    cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255),
                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if keypoints:
        current_eye_position[p] = keypoints[0].pt
    else:
        current_eye_position[p] = None

    if capture_position[p] and current_eye_position[p] is not None:
        keyname = keys[step - 1]

        # Adding the right eye position
        pos = eyes_position.get(keyname, [None, None])
        pos[p] = current_eye_position[p]
        eyes_position[keyname] = pos

        capture_position[p] = False

    return current_eye_position, capture_position, eyes_position

def show_distraction(shown, distractions, width, height):
    # 1.7% chance of adding a new image each frame ~ 30 * 0.017 ~= 1 image every 2 seconds
    if random.random() < 0.017:
        x = random.random() * width
        y = random.random() * height


def main():
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    clock = pygame.time.Clock()

    W, H = pygame.display.Info().current_w, pygame.display.Info().current_h
    initial_radius = min(W, H)
    radius = initial_radius

    # display the backbuffer
    pygame.display.flip()

    # First need the classifiers
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

    # Blob detector
    detector = setup_detector()

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('jeu')
    cv2.createTrackbar('threshold', 'jeu', 71, 255, lambda x: 0)

    eyes_position = dict()  # Dictionnary to store the eyes positions (top, middle, bottom, right and left)
    current_eye_position = [None, None]

    step = 0  # Initialization variable (where the user has to look or if the game starts)
    keys = ['middle', 'top', 'right', 'bottom', 'left']
    capture_position = [False, False]

    thresholds = {
        'top': 0,
        'bottom': 0,
        'right': 0,
        'left': 0
    }

    while True:
        ret, frame = cap.read()
        face_frame = detect_face(frame, face_cascade)

        if face_frame is not None:
            left_eye, right_eye = detect_eyes(face_frame, eye_cascade)
            threshold = cv2.getTrackbarPos('threshold', 'jeu')
            current_eye_position = [None, None]  # Default value

            if left_eye is not None:
                current_eye_position, capture_position, eyes_position = get_values_from_eye(left_eye, blob_process,
                                                                                            detector,
                                                                                            threshold,
                                                                                            current_eye_position,
                                                                                            capture_position,
                                                                                            keys,
                                                                                            step, eyes_position,
                                                                                            "left")

            if right_eye is not None:
                current_eye_position, capture_position, eyes_position = get_values_from_eye(right_eye, blob_process,
                                                                                            detector,
                                                                                            threshold,
                                                                                            current_eye_position,
                                                                                            capture_position,
                                                                                            keys,
                                                                                            step, eyes_position,
                                                                                            "right")

        cv2.imshow('jeu', frame)

        # Analysis done pygame part
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                cv2.destroyAllWindows()
                exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.unicode == 'q':
                    cap.release()
                    cv2.destroyAllWindows()
                    print(eyes_position)
                    exit(0)
                if event.key == pygame.K_SPACE and step <= 4 and not any(capture_position):
                    capture_position = [True, True]  # Left eye and right eye positions must be captured

                    step += 1

        # Get eyes position when looking mid, top, right, bottom and left
        if not any(capture_position) and 0 <= step <= 4:
            setup_crosses(step, screen, W, H)

        # Now all crosses have been drawn we can play
        if step == 6:
            radius = get_new_radius(initial_radius, radius, thresholds, current_eye_position, eyes_position["middle"])
            pygame.draw.circle(screen, RED, (W // 2, H // 2), int(radius))
            pygame.draw.circle(screen, BLACK, (W//2, H//2), 5)
        elif step == 5 and not any(capture_position):
            # All positions have been captured now is type to calculate the thresholds
            x = 0
            y = 1
            left = 0
            right = 1
            # For the top threshold I chose the min
            thresholds['top'] = min(eyes_position['top'][left][y] -
                                    eyes_position['middle'][left][y],
                                    eyes_position['top'][right][y] -
                                    eyes_position['middle'][right][y])

            # Same for bottom but it should be negative so I take the max
            thresholds['bottom'] = max(eyes_position['bottom'][left][y] -
                                       eyes_position['middle'][left][y],
                                       eyes_position['bottom'][right][y] -
                                       eyes_position['middle'][right][y])

            # The difference
            thresholds['right'] = eyes_position['right'][right][x] - \
                                  eyes_position['middle'][right][x]

            thresholds['left'] = eyes_position['left'][left][x] - \
                                 eyes_position['middle'][left][x]
            step += 1

        pygame.display.flip()  # Updating screen

        clock.tick_busy_loop(30)  # Limits framerate to 30 fps

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
