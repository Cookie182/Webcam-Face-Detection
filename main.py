import cv2
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--circle', help='drawing circle around face (default shape is rectange)', action='store_true')
args = parser.parse_args()

sys.path.append(os.path.join(*os.path.abspath(__file__).split("\\")[:-1]))

# pre trained data on face recognition using haar cascade algorithm
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

webcam_image = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    frame_confirm, frame = webcam_image.read()
    if frame_confirm:
        frame = cv2.flip(frame, 1)

        # converting to grayscale for algorithm
        grayscaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_coordinates = face_detector.detectMultiScale(grayscaled_image,
                                                          scaleFactor=1.2, minNeighbors=2, minSize=(60, 60),
                                                          flags=cv2.CASCADE_SCALE_IMAGE)

        # drawing rectangle around detected faces
        for (x, y, width, height) in face_coordinates:
            if not args.circle:
                cv2.rectangle(frame, pt1=(x, y), pt2=(x + width, y + height), color=(0, 0, 225), thickness=4)
            else:
                cv2.circle(frame, center=(x + width // 2, y + height // 2), radius=int(height // 1.8), color=(0, 0, 225), thickness=4)

        # Viewing the Video Feed
        cv2.imshow("Face detection", frame)

        # press Q to quit video
        key = cv2.waitKey(1)
        exit_keys = [ord('q'), ord('Q')]
        if key in exit_keys:
            break
webcam_image.release()
