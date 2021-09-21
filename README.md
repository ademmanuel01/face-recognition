
## Project 1: Face Recognition Library

Python face-recognition library is a simple, user-friendly library with methods useful for to recognize and manipulate faces from Python. Face recognition is a branch of Artificial Intelligence technology which deals about detecting a human face in an image, recognize the identity, and more attributes of a person.

 ![enter image description here](https://img.shields.io/badge/pypi-v3.0.0-green)
 ![enter image description here](https://img.shields.io/badge/py-jupyter-pink) 
 ![enter image description here](https://img.shields.io/badge/pip-dlib-white)  
 ![enter image description here](https://img.shields.io/badge/pip-cmake-white)
 ![enter image description here](https://img.shields.io/badge/pip-opencv-white)  


# Installation

<code> pip </code>

> **pip install face-recognition**,

> **pip install dlib**,

> **pip install dlib**,

> **pip install cmake**,

> **pip install opencv-python**

<code> jupyter notebook </code>
> **python -m pip install jupyter**



## Main Features

* Face_locations() method detects all human faces in the image. Each face is detected as a rectangular frame in the form of a tuple (top,left,bottom,right). 
* Detecting faces and locating the rectangular frames using HOG (Histogram Oriented Gradient) Approach. This method is faster but less accurate. 
* Detecting faces and locating the rectangular frames using Deep Learning based Convolution Neural Network (CNN) Approach. CNN is more accurate but it takes more time to compute. 
* Image proessing. These include - 
  * Locate Faces and Mark with rectangle
  * Writing text on a Face Image
  * Face encoding
  * Distance Function and Resemblance of Faces
  * Face mapping
  * Face compare
* Image Data storage and compare using Python Pandas CSV File
* Attendance Recording in a File
* Image capture using opencv, Time and Date Recording, and Playing audio file from the database

## Usage

### [Application] Project Work - Employee Attendance Management System
##### (https://github.com/ademmanuel01/face-recognition/blob/master/face_recog.ipynb)



Using the main **building_features** function

**building_features**  
* **CMake**
* **Dlib**
* **Open CV**.


*CMake: CMake is a cross-platform free an open source software tool. This is used to manage the software building process using compiler independent method*.

*Dlib: is a dynamic library. This is actually a modem C++ to solve real life problem. This contains machine learning algorithms and tools for building complex software in C++ to solve real life problem. Most of the Machine Learning packages are built on Dlib*.

*Open CV (opensource computer vision) This is a very popular opensource library implementing Computer Vision algorithms using Machine Learning*.

![enter image description here](https://postimg.cc/3WPg0Myq/Screenshot 2021)
 


### Project Work - Employee Attendance Management System
In this project work, I develop an end-to-end attendance management system that uses Face Recognition.

The following are the key modules:
* Reference Data load Module
* Face Capture and Store temporarily
* Face Recognition
* Attendance Record Module
* Display Attendance Module
* Announce Attendance Module

* **Reference Database**: When employees are recruited and join duty as a new employee, the company records the following and store them in the reference database.
1. Employee Number
2. First Name of Employee
3. Last Name of Employee
4. For every employee, his photo is stored, and the location of the photo file is saved as a string.
5. When Attendance is recorded, it will announce the name of the employee. The audio, announcing the name of the employee is stored, and the location of the photo file is saved as a string. 
I stored these employee data in a CSV file.

* **Face Capture Module**: When the employee poses in front of the camera, it captures the image of the face. As the image capture happens continuously, I capture the image frames continuously for 10 frames and then choose the middle one 5th frame. The approach is as follows:
1. Get the camera ready
2. Capture 10 frames continuously one after the other and store each of the 10 frames in the local disk with file names employee 0, employee 1, employee 2………, employee 9, in .jpeg format.
3. Load employee5.jpeg as ukwn (for unknown)
4. Proceed to Face Recognition to recognize the image ukwn


* **Face Recognition Module**: Now we have the following:
1. Face encodings of all the employees of the company in the form of a Python list emp_encod[]
2. Encodings of captured image ukwn: The photo of the employee who posed in front of the camera is captured in a Python variable ukwn. This photo must be first encoded as ukwn_encode
3. Note: A word of caution. As ukwn is an image captured in the camera, it is possible that the quality is poor. In such a case the face in the image may not be visible and encodings cannot be computed. So, it is recommended that we give the encodings command be given in a try: except clause hence the Exception is created, printed, and exit. 

* **Record Attendance (in a Datafile) Module**: The attendance in a data file in text format is recorded. The name of the file is Attendance.txt. The attendance is recorded in a single line with four fields as one string as follows:
1.Emp No | 2. First Name | 3.Last Name | 4.Data time stamp. 

* **Display Attendance Recorded Module**: After recognizing the face, then display the name of the person. On the image captured in the camera, the employee's name to be written and displayed. If the face is not recognized, it will display "Face NOT Recognized"

* **Announce (audio) Module**: The audio file announcing the name of the person is loaded as an audio file and the path is given in the Python array audiolocation[]. If i is the index, it must play audiolocation[i].

```python
from PIL import Image, ImageDraw, ImageFont
import face_recognition
import cv2
import sys
import pandas as pd
import datetime
import pygame

emp.append(face_recognition.load_image_file(photolocation[i]))
emp_encod.append(face_recognition.face_encodings(emp[i])[0]) 
cv2.imwrite('Employee'+str(i)+'.png', image)
pygame.mixer.music.load(audioloc)
pygame.mixer.music.play()
    
```


## Dependencies

* face_recognition
* pandas 
* datetime 
* pygame
* opencv
* PIL
* numpy 


### Source Code

<code> You can get the latest source code </code>

> git clone https://github.com/ademmanuel01/face-recognition/blob/master/face_recog.ipynb 
