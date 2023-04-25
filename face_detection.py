import os
import cv2
import face_recognition
import shutil
from deepface import DeepFace
import pyttsx3
import time

n = 0
# Load the known faces and their names
known_faces = []
known_names = []

# Get a list of all image files in the folder
image_folder = "path to your database of images "
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Loop through each image file and encode the faces
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    img = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(img)
    if len(face_encodings) > 0:
        face_encoding = face_encodings[0]
        known_faces.append(face_encoding)
        known_names.append(os.path.splitext(image_file)[0])
num_faces_detected = 0
# Initialize the webcam
video_capture = cv2.VideoCapture(0)
# Initialize text-to-speech engine
engine = pyttsx3.init()
while n < 3:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through each face
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = frame[y:y+h, x:x+w]
        face_region = frame[y:y+h, x:x+w]

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)

        name = "Unknown"

        # If a match was found in known_faces, use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # Get gender, age, and emotion category using DeepFace
        demography = DeepFace.analyze(face_region, actions=['gender', 'emotion', 'age'], enforce_detection=False)

        # Get the results for the first face detected
        result = demography[0]

        # Get the gender, age, and dominant emotion
        gender = result['gender']
        age = int(result['age'])
        emotion = result['dominant_emotion']

        # Convert gender value to string
        if gender['Man'] > 50:
            gender_str = "man"
        else:
            gender_str = "woman"


        print("Name: ", name)
        print("Gender: ", gender_str)
        print("Age: ", age)
        print("Emotion: ", emotion)
        engine.say(f'I can see a {emotion} {gender_str} person who looks {age} years old and might be {name}')
        print(f'I can see a {emotion} {gender_str} person who looks {age} years old and might be {name}')
        engine.runAndWait()
     


        # Move the current face image to a folder with the detected name
        if name != "Unknown":
            detected_folder = os.path.join("detected_faces", name)
            if not os.path.exists(detected_folder):
                os.makedirs(detected_folder)
            file_name = str(len(os.listdir(detected_folder)) + 1) + ".jpg"
            file_path = os.path.join(detected_folder, file_name)
            shutil.copy(image_path, file_path)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name, gender, age, and emotion category below the face
        label = name 
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
    # Display


    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q


    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        n = n+1
        cv2.destroyAllWindows()
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
