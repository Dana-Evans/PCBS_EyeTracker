"""
The main program that is used to play the game.

:author: `Dana Ladon <dana.ladon@ens.fr>`_

:date:  2019, december

"""
import os
import pickle
import random
from datetime import datetime, timedelta

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


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


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
    """
    A function to draw cross with position depending on the step.

    :param step: The step, where we are in time
    :type step: int
    :param screen: The screen where I have to draw the cross
    :type screen: pygame.Surface
    :param W: The width of the screen
    :type W: int
    :param H: The height of the screen
    :type H: int
    """
    if step == 0:
        # Middle cross
        draw_cross(screen, W // 2, H // 2, 50, 50)
    elif step == 1:
        # Top cross
        draw_cross(screen, W // 2, H // 10, 50, 50)
    elif step == 2:
        # Right cross
        draw_cross(screen, 9 * W // 10, H // 2, 50, 50)
    elif step == 3:
        # Bottom cross
        draw_cross(screen, W // 2, 9 * H // 10, 50, 50)
    elif step == 4:
        # Left cross
        draw_cross(screen, W // 10, H // 2, 50, 50)


def setup_detector():
    """
    A function to create and setup a blob detector.

    :return: The simple blob detector set up.
    :rtype: cv2.SimpleBlobDetector
    """
    detector_params = cv2.SimpleBlobDetector_Params()
    detector_params.filterByArea = True
    detector_params.maxArea = 1500
    detector_params.filterByConvexity = False
    detector_params.filterByInertia = False
    return cv2.SimpleBlobDetector_create(detector_params)


def get_new_radius(initial_radius, radius, thresholds, position,
                   center_position):
    """
    Return new radius depending on where the user is looking.

    :param initial_radius: The initial radius to return if the user is looking somewhere else than in the center
    :type initial_radius: int
    :param radius: The actual radius
    :type radius: int
    :param thresholds: All the thresholds for every position, if the dx > threshold the user is not looking in the
                       center
    :type thresholds: dict
    :param position: The position of the eyes
    :type position: list of tuple of two int
    :param center_position: The position of the eyes when looking at the center
    :type center_position: list of tuple of two int
    :return: The new radius
    :rtype: int
    """
    if not all(position):
        return radius

    fps = 30
    # It shoud take around 10 sec to totally reduce the circle
    dr = initial_radius // (10 * fps)

    dx_right = position[1][0] - center_position[1][0]
    dx_left = position[0][0] - center_position[0][0]
    # dy_right = position[1][1] - center_position[1][1]
    # dy_left = position[0][1] - center_position[0][1]

    # Cannot take into account top and bottom because it is too sensitive
    if dx_right < thresholds['right'] or \
            dx_left > thresholds['left']:
        return initial_radius

    return radius - dr


def get_values_from_eye(eye, detector, threshold, current_eye_position, eye_name):
    """
    Return the new values for the current eyes position, if the capture have been done and the eyes position list.

    :param eye: The eye we are focusing on
    :type eye: pygame.ndarray
    :param detector: The blob detector we are using
    :type detector: cv2.SimpleBlobDetector
    :param threshold: The threshold for the eyes (to convert in black and white only)
    :type threshold: int
    :param current_eye_position: The current eye position array to update if necessary
    :type current_eye_position: list of tuple of 2 ints
    :param eye_name: the eye name (can be left or right)
    :type eye_name: str
    :return: The updated value of the current_eye_position
    :rtype: list of tuples of 2 ints
    """
    p = 1 if eye_name == "right" else 0

    eye = cut_eyebrows(eye)
    keypoints = blob_process(eye, detector, threshold)
    cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255),
                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if keypoints:
        current_eye_position[p] = keypoints[0].pt

    return current_eye_position


@static_vars(shown=[],
             distractions=[])
def show_distractions(width, height, screen):
    """
    A function to show distractions while the user tries to reduce the center.

    :param screen: The screen where the images are shown
    :type screen: pygame.Surface
    :param width: The width of the screen
    :type width: int
    :param height: The height of the screen
    :type height: int
    """
    # 6.66% chance of adding a new image each frame ~ 30 * 0.0666 ~= 2 image every second
    if random.random() < 0.0666:
        x = int(random.random() * width)
        y = int(random.random() * height)
        distraction = random.choice(show_distractions.distractions)
        show_distractions.shown.append((x, y, distraction))

    if len(show_distractions.shown) > 2:
        show_distractions.shown.pop(0)

    for x, y, image in show_distractions.shown:
        screen.blit(image, (x, y))


def main():
    """ The main function that is used when the script starts. """
    # Pygame initialization
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    myfont = pygame.font.SysFont('Arial', 30)

    # Clock
    clock = pygame.time.Clock()

    # Before anything shows the information screen
    information_screen(screen, myfont, clock)

    # Load and scale images
    show_distractions.distractions = [pygame.image.load(os.path.join("data", "cats", "png", filepath)).convert_alpha()
                                      for filepath in os.listdir(os.path.join("data", "cats", "png"))]
    show_distractions.distractions = [pygame.transform.scale(image, (64, 64)) for image in
                                      show_distractions.distractions]

    # Game and display infos
    W, H = pygame.display.Info().current_w, pygame.display.Info().current_h
    initial_radius = min(W, H)
    radius = initial_radius

    # Score loading
    scores = []
    scores_file_path = os.path.join("data", "scores")

    if os.path.isfile(scores_file_path):
        with open(scores_file_path, 'rb') as scores_file:
            scores = pickle.load(scores_file)

    # Display the buffer
    pygame.display.flip()

    # First need the classifiers
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

    # Blob detector
    detector = setup_detector()

    # cv2 window setup
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('jeu')
    cv2.createTrackbar('threshold', 'jeu', 71, 255, lambda foo: 0)
    cv2.createTrackbar('Record score', 'jeu', 1, 1, lambda foo: 0)

    # Variables initialization

    eyes_position = dict()
    current_eye_position = [None, None]
    step = 0
    keys = ['middle', 'top', 'right', 'bottom', 'left']
    capture_position = [False, False]
    thresholds = {
        'top': 0,
        'bottom': 0,
        'right': 0,
        'left': 0
    }

    text_well_done = None
    text_best = None
    text_median = None
    text_mean = None

    # The time will be initialized once the game is started not before
    starting_time = 0

    while True:
        ret, frame = cap.read()
        face_frame = detect_face(frame, face_cascade)

        if face_frame is not None:
            left_eye, right_eye = detect_eyes(face_frame, eye_cascade)
            threshold = cv2.getTrackbarPos('threshold', 'jeu')
            current_eye_position = [None, None]  # Default value

            if left_eye is not None:
                current_eye_position = get_values_from_eye(left_eye,
                                                           detector,
                                                           threshold,
                                                           current_eye_position,
                                                           "left")

                if capture_position[0] and current_eye_position[0] is not None:
                    keyname = keys[step - 1]

                    # Adding the right eye position
                    pos = eyes_position.get(keyname, [None, None])
                    pos[0] = current_eye_position[0]
                    eyes_position[keyname] = pos

                    capture_position[0] = False

            if right_eye is not None:
                current_eye_position = get_values_from_eye(right_eye,
                                                           detector,
                                                           threshold,
                                                           current_eye_position,
                                                           "right")

                if capture_position[1] and current_eye_position[1] is not None:
                    keyname = keys[step - 1]

                    # Adding the right eye position
                    pos = eyes_position.get(keyname, [None, None])
                    pos[1] = current_eye_position[1]
                    eyes_position[keyname] = pos

                    capture_position[1] = False

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

        if step == 5 and not any(capture_position):
            # All positions have been captured now is type to calculate the thresholds
            x = 0
            y = 1
            left = 0
            right = 1

            # For the top threshold I chose the max
            thresholds['top'] = max(eyes_position['top'][left][y] -
                                    eyes_position['middle'][left][y],
                                    eyes_position['top'][right][y] -
                                    eyes_position['middle'][right][y])

            # Same for bottom but it should be positive so I take the min
            thresholds['bottom'] = min(eyes_position['bottom'][left][y] -
                                       eyes_position['middle'][left][y],
                                       eyes_position['bottom'][right][y] -
                                       eyes_position['middle'][right][y])

            # The difference
            thresholds['right'] = eyes_position['right'][right][x] - \
                                  eyes_position['middle'][right][x]

            thresholds['left'] = eyes_position['left'][left][x] - \
                                 eyes_position['middle'][left][x]

            starting_time = datetime.now()

            step += 1
        elif step == 6:
            # Thresholds and positions have been captured, last thing is to play !
            # How long will you take ?
            radius = get_new_radius(initial_radius, radius, thresholds, current_eye_position, eyes_position["middle"])

            if radius <= 0:
                step += 1

            # Reducing circle
            pygame.draw.circle(screen, RED, (W // 2, H // 2), int(radius))

            # Center focusing point
            pygame.draw.circle(screen, BLACK, (W // 2, H // 2), 5)

            show_distractions(W, H, screen)

        elif step == 7:
            # Last step, scoring

            delta = datetime.now() - starting_time
            # Bit of a hack to get the total time in microseconds
            delta_time = delta / timedelta(microseconds=1)
            delta_time = round(delta_time / 1e6, 2)
            print(f"Wow you won in {delta_time} seconds !")

            record = cv2.getTrackbarPos('Record score', 'jeu')

            if record == 1:
                scores.append(delta_time)

                with open(scores_file_path, 'wb+') as scores_file:
                    pickle.dump(scores, scores_file)

            text_well_done = myfont.render(f'Well done ! You have managed to finish in {delta_time} seconds !',
                                           False, (100, 255, 100))

            if scores:
                mean = round(sum(scores) / len(scores), 4)
                median = scores[len(scores) // 2]
                best = sorted(scores)[0]
                text_mean = myfont.render(f'The mean is {mean} seconds', False, (0, 255, 0))
                text_median = myfont.render(f'The median is {median} seconds', False, (0, 255, 0))
                text_best = myfont.render(f'The best time is {best} seconds', False, (0, 255, 0))

            step += 1
        elif step == 8:

            if scores:
                total_height = text_well_done.get_rect().height * 4  # height is the same for all text
                screen.blit(text_well_done, (W // 2 - text_well_done.get_rect().width // 2, H // 2 - total_height // 2))
                screen.blit(text_mean, (W // 2 - text_mean.get_rect().width // 2, H // 2 - total_height // 4))
                screen.blit(text_median, (W // 2 - text_median.get_rect().width // 2, H // 2 + total_height // 4))
                screen.blit(text_best, (W // 2 - text_best.get_rect().width // 2, H // 2 + total_height // 2))
            else:
                screen.blit(text_well_done, (W // 2 - text_well_done.get_rect().width // 2, H // 2))

        pygame.display.flip()  # Updates screen

        clock.tick_busy_loop(30)  # Limits framerate to 30 fps

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def information_screen(screen, myfont, clock):
    """
    A first screen to show the instructions.

    :param screen: The screen where to write the instructions
    :type screen: pygame.Suface
    :param myfont: The font used to write
    :type myfont: pygame.font.Font
    :param clock: The clock used to synchronize the framerate
    :type clock: Clock
    """
    W, H = pygame.display.Info().current_w, pygame.display.Info().current_h
    pygame.display.flip()

    loop = True

    while loop:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.unicode == 'q' or event.key == pygame.K_ESCAPE:
                    exit(0)

                if event.key == pygame.K_SPACE or event.key == pygame.K_KP_ENTER:
                    loop = False

        # Show screen information
        info = """
        Welcome to eye tracker, my PCBS project.
        This is a game programmed to improve your focusing.
        How to play :
            - After this window you will have to adjust the eye tracker such that your eyes are well tracked
                The first trackbar allows you to adjust the contrast of your room, you should move it around until your 
                eyes are surrounded with a red circle. If it is too big, reduce it, if it is too small increase it.
                The second trackbar allows you to chose whether you want to save your score or not, 0 means no and 1 
                means yes.
            - Once you have adjusted it, switch with the other window that opened and look at the middle of the blue
                crosses and press the spacebar. This allows to calibrate the eye tracker with your eye's movement.
            - After all the five blue crosses have been checked you will have to focus on the black point in the middle
                of the red circle. Your goal is to keep focused on it. Everytime you look somewhere else the circle will
                return to its original radius. Your goal is to make it disappear.
        To go to the next window press the space bar !
        """

        infos = info.split("\n")

        total_height = 0
        text_info = []
        for text in infos:
            tmp = myfont.render(text, False, (10, 36, 200))
            total_height += tmp.get_rect().height
            text_info.append(tmp)

        n = len(text_info)
        total_height += 10 * n  # Adding ten spacing between lines

        for i in range(n):
            text = text_info[i]
            centered = i - n // 2
            screen.blit(text, (100, H // 2 + centered * total_height // n))

        pygame.display.flip()  # Updates screen

        clock.tick_busy_loop(30)  # Limits framerate to 30 fps


if __name__ == "__main__":
    main()
