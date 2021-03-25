### General imports ###
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

import cv2
import datetime

from time import time
from time import sleep
import re
import os

import argparse
from collections import OrderedDict

### Image processing ###
from scipy.ndimage import zoom
from scipy.spatial import distance
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS
from scipy import ndimage

import dlib

from tensorflow.keras.models import load_model
from imutils import face_utils

import requests
import json
import socket

from oscpy.server import OSCThreadServer
from oscpy.client import OSCClient

global shape_x
global shape_y
global input_shape
global nClasses
trigger = False
done = False

# ====================
# OSC
server = OSCThreadServer()
server.listen(address='localhost', port=4002, default=True)
client = OSCClient('localhost', 5001)
# ====================

# Wait for OSC ready message


def incoming_message(msg):
    global trigger, done

    print("Message from Headlines APP!")

    if msg == True:
        trigger = True
        client.send_message(b'/emo-running', [trigger])
    else:
        trigger = False
        done = True

# Main emotion extraction function


def show_webcam():
    global trigger, done

    shape_x = 48
    shape_y = 48
    input_shape = (shape_x, shape_y, 1)
    nClasses = 7

    thresh = 0.25
    frame_check = 20

    def eye_aspect_ratio(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_face(frame):

        # Cascade classifier pre-trained model
        cascPath = '../Models/Landmarks/face_landmarks.dat'
        faceCascade = cv2.CascadeClassifier(cascPath)

        # BGR -> Gray conversion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Cascade MultiScale classifier
        detected_faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                                      minSize=(
                                                          shape_x, shape_y),
                                                      flags=cv2.CASCADE_SCALE_IMAGE)
        coord = []

        for x, y, w, h in detected_faces:
            if w > 100:
                sub_img = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 1)
                coord.append([x, y, w, h])

        return gray, detected_faces, coord

    def extract_face_features(faces, offset_coefficients=(0.075, 0.05)):
        gray = faces[0]
        detected_face = faces[1]

        new_face = []

        for det in detected_face:

            # X and Y is the gray conversion; w, h is height and width
            x, y, w, h = det

            # Offset coefficient, np.floor takes the lowest integer (delete border of the image)
            horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
            vertical_offset = np.int(np.floor(offset_coefficients[1] * h))

            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # gray transforms the image
            extracted_face = gray[y+vertical_offset:y+h,
                                  x+horizontal_offset:x-horizontal_offset+w]

            # Zoom on the face
            new_extracted_face = zoom(
                extracted_face, (shape_x / extracted_face.shape[0], shape_y / extracted_face.shape[1]))
            # cast type float
            new_extracted_face = new_extracted_face.astype(np.float32)
            # scale
            new_extracted_face /= float(new_extracted_face.max())
            # print(new_extracted_face)

            new_face.append(new_extracted_face)

        return new_face

    # Facial landmarks
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

    (eblStart, eblEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (ebrStart, ebrEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

    model = load_model('../Models/EmotionXCeption/video.h5')
    face_detect = dlib.get_frontal_face_detector()
    predictor_landmarks = dlib.shape_predictor(
        "../Models/Landmarks/face_landmarks.dat")

    # Video capture
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    video_capture.set(cv2.CAP_PROP_FPS, 2)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 1)
    #ret_val, img = video_capture.read()
    #cv2.imshow('my webcam', img)

    if (video_capture.isOpened()):
        print("ready!")

    # JSON dictionary
    face_record = {}
    face_record['frames'] = []

    # Wait of news screens OSC and trigger emotion analysis
    server.bind(b'/emotions', incoming_message)

    while True:

        if trigger:  # Analyse face if news screen is playing and human in front

            print("clearing old data...")
            open('emotions.json', 'w').close()  # detele previous data

            frameCount = 0

            while trigger:
                # Capture frame-by-frame
                ret, frame = video_capture.read()
                width = 320
                height = 240
                frame = imutils.resize(frame, width, height)

                if not ret:
                    break

                #gray, detected_faces, coord = detect_face(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = face_detect(gray, 1)

                for (i, rect) in enumerate(rects):
                    #print("frame captured")
                    if i == 0:  # take only the first face

                        # Find face
                        shape = predictor_landmarks(gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        # Identify face coordinates
                        (x, y, w, h) = face_utils.rect_to_bb(rect)
                        face = gray[y:y+h, x:x+w]

                        # Try to zoom into the face
                        try:
                            # Zoom on extracted face
                            face = zoom(
                                face, (shape_x / face.shape[0], shape_y / face.shape[1]))

                        # Weird interpolation error upon entering the frame
                        except ZeroDivisionError:
                            pass

                        try:
                            # Cast type float
                            face = face.astype(np.float32)

                            # Scale
                            face /= float(face.max())
                            face = np.reshape(face.flatten(), (1, 48, 48, 1))

                            # Make Prediction
                            prediction = np.float64(model.predict(face))
                            prediction_result = np.argmax(prediction)

                            # wait for next frame which should be fine
                        except ValueError:
                            pass

                        # Add new emotion percentiles to dictionary
                        emotions = {
                            "angry": round(prediction[0][0], 3),
                            "disgusted": round(prediction[0][1], 3),
                            "afraid": round(prediction[0][2], 3),
                            "happy": round(prediction[0][3], 3),
                            "sad": round(prediction[0][4], 3),
                            "surprised": round(prediction[0][5], 3),
                            "neutral": round(prediction[0][6], 3)
                        }

                        # Append values to JSON
                        face_record['frames'].append({
                            'emotion-probs': emotions
                        })

                        frameCount += 1

                        if frameCount > 1:
                            print("frame:", frameCount-1)
                        else:
                            print('Dropping useless frame')

            if done:
                trigger = False
                done = False

                # drop the first frame which
                # is usually faulty if there is a frame at all
                if len(face_record['frames']) > 0:
                    face_record['frames'].pop(0)

                # Write frames to JSON
                with open("emotions.json", "a+", encoding='utf-8') as outfile:
                    json.dump(face_record, outfile,
                              ensure_ascii=False, indent=4)

                # send message to Fito
                client.send_message(b'/emo-done', [True])
                print('done!')

                # JSON dictionary
                face_record = {}
                face_record['frames'] = []

    # When everything is done, release the capture
    video_capture.release()


def main():
    try:
        show_webcam()
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
