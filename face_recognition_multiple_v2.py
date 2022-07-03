# face_recognition_multiple.py ver2.0
# face recognition using /known and /unknown directory of images
# feed this an unknown image and will go through the /known faces to match and label the unknown image
# pre-process all known images and save into memory all face encodings
# 7.3.22

import time
import cv2
import face_recognition
import os

start_time = time.time()

# List of known images
known_images_path = "./known_faces"
known_images_filenames = os.listdir(known_images_path)
print(known_images_filenames)

# pre-process all known faces in our library
face_encodings = {}     # Dictionary Object  'face_label' : 99999
for known_image_filename in known_images_filenames:
    print('pre-processing and encoding ' + known_image_filename)
    known_image = face_recognition.load_image_file(known_images_path + '/' + known_image_filename)
    face_label = os.path.splitext(os.path.basename(known_image_filename))[0]    # i.e. the filename
    # store key:value pair to dictionary
    face_encodings[face_label] = face_recognition.face_encodings(known_image)[0]

# our unknown photo, may contain single or multiple faces
unknown_image = face_recognition.load_image_file('./unknown_faces/gallery1.jpg')

# find faces in unknown image
face_locations_unknown_image = face_recognition.face_locations(unknown_image)

# for each face found in the unknown image, run it through our known images directory to try and identify person
for face_location in face_locations_unknown_image:
    # get location of current face found in unknown image
    top, right, bottom, left = face_location
    # extract the face (region of interest)
    unknown_face_image = unknown_image[top:bottom, left:right]   # height, width
    unknown_face_encoding = face_recognition.face_encodings(unknown_face_image)[0]   # encode unknown face
    # loop through all pre-processed face encodings and compare against the unknown image
    for face_encoding in face_encodings:
        # display progress status
        print('.', end='')
        # check if we have a match for this particular face
        # but first, we need to get actual value from dictionary based on current key (face_encoding)
        match = face_recognition.compare_faces(unknown_face_encoding, [face_encodings.get(face_encoding)])
        if match[0]:
            face_label = face_encoding  # the current dictionary key is the matching person's name
            # draw box and label around face
            cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 255, 0), 2)
            # label image
            cv2.rectangle(unknown_image, (left, top), (right, top - 50), (0, 255, 0), -1)
            cv2.putText(unknown_image, face_label, (left + 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4)
            break  # exit this loop since we already found a match
        else:
            face_label = "Unknown"
    # we've processed all face_encodings and there was no match for this particular face
    # So label this face as 'Unknown'
    cv2.rectangle(unknown_image,(left,top), (right, bottom), (0,255,0), 2)
    cv2.rectangle(unknown_image, (left, top), (right,top-50), (0,255,0), -1)
    cv2.putText(unknown_image, face_label, (left+10,top-10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0),4)
    # after we go through all the known face encodings for this particular face, process the next face...

end_time = time.time()
print("\nProcessing took {:2f} seconds".format(end_time-start_time))

# display results!
cv2.imshow('Output', cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)

cv2.destroyAllWindows()


