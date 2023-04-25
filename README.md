# Audio-Based-Face-Identity-Detection-using-Deepface&Face_Recognition

This project uses a combination of computer vision and machine learning techniques to detect and recognize faces in real-time from video feed captured using a webcam. The system uses OpenCV's face detection, Deepface library for face analysis, and Face Recognition for facial recognition. It also uses pyttsx3 library for text-to-speech output.

Prerequisites
Python 3.x
OpenCV 4.x
face_recognition 1.2.3
deepface 0.0.68
pyttsx3 2.90

Installation
1. Clone the repository:
```git clone https://github.com/Cdbrain786/Audio_Based-Face-Identity-Detection-using-Deepface-Face_recognition.git```
   
cd Audio-Based-Face-Identity-Detection-using-Deepface-Face_Recognition

2. Install the required libraries:
  ``` pip install -r Requirement.text```
   
Usage
1. Prepare a folder containing the images of the people whose faces you want to recognize. Make sure each image file is named after the person's name.

2. Change the image_folder variable in face_detection.py to the path of your image folder.

3. Run the script.```python face_detection.py```
   
4. When the script starts, it will capture video feed from your webcam and detect faces in real-time. If a face is detected, it will compare it to the known faces in   the image folder and try to recognize it. It will also analyze the face using Deepface and provide information about the person's gender, age, and dominant emotion.

5. The script will output the recognized person's name, gender, age, and emotion as speech using pyttsx3 library. It will also draw a box around the detected face and display the label with the person's name, gender, age, and emotion below the face.

6. If the recognized person's name is not "Unknown", the script will move the current face image to a folder with the detected name.

7. Press 'q' to exit the script.

License
This project is licensed under the MIT License.

Acknowledgements

OpenCV
Deepface
Face Recognition
pyttsx3
