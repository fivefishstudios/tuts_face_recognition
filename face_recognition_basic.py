# face_recognition_basic.py
# basic face recognition using /known and /unknown directory of images
# this program is very basic. If the unknown image has multiple faces, it cannot detect the other unknown faces in the photo
# Also, the logic of this program is using a known face and seeing if it can identify the unknown face and it matches the known face
# a more practical use will be feed the program an unknown photo, and try to match all faces in the unknown photo using the /known directory photos
# 7.2.22

import cv2
import face_recognition
import os

known_filepath = './known_faces/Erin Oquindo.jpg'
known_image = face_recognition.load_image_file(known_filepath)
person_name = os.path.splitext(os.path.basename(known_filepath))[0]

unknown_image = face_recognition.load_image_file('./unknown_faces/erin-play.jpg')

known_image_encoding = face_recognition.face_encodings(known_image)[0]    # get 1st item
unknown_image_encoding = face_recognition.face_encodings(unknown_image)[0]      # get 1st item

# find faces in unknown image
face_locations = face_recognition.face_locations(unknown_image)

# returns True or False
results = face_recognition.compare_faces(unknown_image_encoding, [known_image_encoding])

print("results: ", results )


for face_location in face_locations:
    if results[0]:
        img_label = person_name
    else:
        img_label = "Unknown"
    # annotate the unknown image
    top, right, bottom, left = face_location
    cv2.rectangle(unknown_image,(left,top), (right, bottom), (0,255,0), 2)
    # label image
    cv2.rectangle(unknown_image, (left, top), (right,top-50), (0,255,0), -1)
    cv2.putText(unknown_image, img_label, (left+10,top-10), cv2.FONT_HERSHEY_SIMPLEX,1.3, (0,0,0),4)


cv2.imshow('Output', cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)