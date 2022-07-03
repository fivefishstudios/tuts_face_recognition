# face_recognition_train_v1.py
# batch training for face recognition using /known_faces directory
# 7/3/22

import time
import cv2
import face_recognition
import os
import pickle5 as pickle

start_time = time.time()

# List of known images
# v1: process all images
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
    face_encoding = face_recognition.face_encodings(known_image)
    if face_encoding:
        face_encodings[face_label] = face_recognition.face_encodings(known_image)[0]
    else:
        print("!!! unrecognized face, skipping " + face_label)

end_time = time.time()

# save face_encodings to disk for faster recall
file = open("Face-Training-Data.pkl", "wb")
pickle.dump(face_encodings, file)
file.close()

print("successfully written Face-Training-Data.pkl ")
print("batch processing took {:0.2f} seconds".format(end_time - start_time))







# =================================================
# TRY THIS, STORING DICTIONARY TO AN EXTERNAL FILE
# 4. Pickle.dump()
# First, open the file in write mode by using “wb”, this mode is used to open
# files for writing in binary format.Use pickle.dump() to serialize dictionary data and then write to file.
#
# To read a file we need to open the binary file in reading
# mode(“rb”), then use the pickle.load() method
# to deserialize the file contents. To check the output we are Printing
# the file contents using the print() method.

# Program
# # python 3 program to write and read dictionary to text file
#
# import pickle
#
# dict_students = {'Name': 'Jack', 'Sub': 'Math', 'marks': 100, 'Grade': 'A'}
# file = open("DictFile.pkl", "wb")
# pickle.dump(dict_students, file)
# file.close()
#
# # reading the DictFile.pkl" contents
# file = open("DictFile.pkl", "rb")
# file_contents = pickle.load(file)
# print(file_contents)
#
# Output:
# The file will get created in the current directory with the following data format.
# {'Name': 'Jack', 'Sub': 'Math', 'marks': 100, 'Grade': 'A'}

