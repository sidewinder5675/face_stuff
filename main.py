#!/usr/bin/env python3
import face_recognition
import cv2
import numpy as np
import os

photos_path = '/Users/Todd/face/Project/photos'

print (photos_path)
# Function to generate dict of people
def get_people_dict(path):
    # declare variables
    l = list()
    d = dict()
    
    # gather everthing in path given
    for dir, folders, files in os.walk(path):
        l.append([dir,folders,files])
    
    # loop over results
    for i, item in enumerate(l):
        # skip first array, root dir garbo
        if i != 0:
            # add a dict entry to d for every person as key
            # and all photos as value
            name = os.path.basename(item[0])
            
#            manager = os.path.basename(item[0])
            photos = list()
            for photo in item[2]:
                photos.append(os.path.join(item[0],photo))
            d[name]=photos
    return d

# Function to update arrays of known face encodings and their names
def init_known_face_lists(people_dict):
    # define empty lists to be returned
    known_face_encodings = list()  # list of photo paths
    known_face_names =  list()  # list of names (Todd Seydel, David Cooney)

    # loop over people
    for name in people:
        photo = people[name][0]
        encoding = photo
        
        # use face_recognition to create encodings
        image = face_recognition.load_image_file(photo)
        encoding = face_recognition.face_encodings(image)[0]

        # add results to appropriate lists
        known_face_encodings.append(encoding)
        known_face_names.append(name)
    return known_face_encodings, known_face_names


# Get People
people = get_people_dict(photos_path)  ## path and list of names
print (people)

# Update known_face lists
known_face_encodings,known_face_names = init_known_face_lists(people)
print (known_face_names)

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


while True:

    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)


    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
#        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 1)

        # Draw a label with a name below the face
#        cv2.rectangle(frame, (left - 1, bottom - 10), (right - 1, bottom - 1), (40, 40, 40), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        
        cv2.putText(frame, name, (right - 60, top - 50), font, .8, (255, 255, 255), 1)
#        cv2.putText(frame, name_manager, (right - 60, top - 6), font, .5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
