#!/usr/bin/env python
# coding: utf-8

# ## Face Recognition
# Face recognition is a branch of Artificial Intelligence technology which deals about detecting a human face in an image, recognize the identity, gender, age and more attributes of the person and even the emotions.
# Face recognition is one of the high-level intellectual capability of human beings
# 
# ## Face Recognition Approaches
# Broadly there are three sets of scientists working on face recognition and
# related fields. They are
# 1. Medical Scientists – Cognitive strategies
# 2. Anthropological Researchers
# 3. Artificial Intelligence Researchers using Computer Vision with Deep Learning [Convolution Neutral Networks].
# 
# In Artificial Intelligence research the image of the face is represented as a matrix of pixels. By analyzing the image, we try to find the unique set of numbers, called a signature. We use face signature for face recognition. Each image is  considered as a matrix of pixels. Each pixel is represented by a set of numbers.

# ## Python face-recognition Library
# 
# Python face-recognition library is a simple, user-friendly library with methods useful for to recognize and manipulate faces from Python
# 
# ## Python library Face-recognition is built on three important foundations.
# 1. CMake
# 2. Dlib
# 3. Open CV
# 
# CMake: CMake is a cross-platform free an open source software tool. This is used to manage the software building process using compiler independent method.
# 
# Dlib: is a dynamic library. This is actually a modem C++ to solve real life problem. This contains machine learning algorithms and tools for building complex software in C++ to solve real life problem. Most of the Machine Learning packages are built on Dlib
# 
# Open CV (opensource computer vision) This is a very popular opensource library implementing Computer Vision algorithms using Machine Learning.
# 
# 

# In[1]:


#loading of images and check for the image shape

import face_recognition
photo1 = face_recognition.load_image_file('./pics&group/pic1.jpeg','RGB')
photo2 = face_recognition.load_image_file('./pics&group/pic2.jpeg','RGB')
photo3 = face_recognition.load_image_file('./pics&group/pic3.jpeg')
print("The shapes of the Color photo files are...")
print("photo1 Photo....",photo1.shape)
print("photo2 Photo.....", photo2.shape)
print("photo3 Photo......", photo3.shape)


# In[2]:


# we want to load the files in Black and White format we must use using 'L'
# attribute in the second attribute of load_image_file() function as follows:

photoA = face_recognition.load_image_file('./pics&group/pic1.jpeg','L')
photoB = face_recognition.load_image_file('./pics&group/pic2.jpeg','L')
photoC = face_recognition.load_image_file('./pics&group/pic3.jpeg', 'L')
print("The shapes of the bw photo files are...")
print("photoA Photo....",photoA.shape)
print("photoB Photo.....", photoB.shape)
print("photoC Photo......", photoC.shape)


# ## Python Image Library
# 
# Python Image Library (PIL) is a opensource library for Python that supports image processing
# activities of pictures, including the display

# In[3]:


from PIL import Image
import face_recognition
pil_image1 = Image.fromarray(photo1)
pil_image1.show()
pil_image2 = Image.fromarray(photo2)
pil_image2.show()
pil_image3 = Image.fromarray(photo3)
pil_image3.show()


# In[4]:


pil_image1 = Image.fromarray(photoA)
pil_image1.show()
pil_image2 = Image.fromarray(photoB)
pil_image2.show()
pil_image3 = Image.fromarray(photoC)
pil_image3.show()


# ## Face Locations Method
# 
# In python face-recognition library, face_locations() method detects all human faces in the image.
# Each face is detected as a rectangular frame in the form of a tuple (top,left,bottom,right). If there are n faces, the output is a list of n tuples with four entries as (top, right, bottom, left).

# ## HOG (Histogram Oriented Gradient) Approach.
# 
# In detecting the faces and locating the rectangular frames HOG (Histogram Oriented Gradient) Approach. This is faster but less accurate.

# In[5]:


from PIL import Image
import face_recognition
test_pic3 = face_recognition.load_image_file('./pics&group/pic3.jpeg')
print(test_pic3.shape)
l = face_recognition.face_locations(test_pic3, model = 'hog')
print(l)
top = l[0][0]
right = l[0][1]
bottom = l[0][2]
left = l[0][3]
pic3_face = test_pic3[top:bottom, left:right]
print(test_pic3.shape)
pic3_face_image = Image.fromarray(pic3_face)
pic3_face_image.show()


# In[2]:


#Face Detection using HOG Model 

from PIL import Image
import face_recognition

# Load the jpg file into a numpy array
group_photo = face_recognition.load_image_file("./pics&group/group1.jpeg")

# Find all the faces in the image 
face_locations = face_recognition.face_locations(group_photo, model="hog")
# Let us print the number of faces in the Photo
print("There are  {} face(s) in this photo".format(len(face_locations)))
face_count = 0
for face_location in face_locations:
    # Print the location of each face in this image
    face_count = face_count+1
    top, right, bottom, left = face_location
    print("Face {}...Top: {}, Left: {}, Bottom: {}, Right: {}".format(face_count,top, left, bottom, right))
    # You can access the actual face itself like this:
    face_image = group_photo[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()


# ## Deep Learning based Convolution Neural Network (CNN) Approach.
# 
# CNN is more accurate but it takes more time to compute.

# In[3]:


from PIL import Image
import face_recognition
test2_photo = face_recognition.load_image_file('./pics&group/pic3.jpeg')
print(test2_photo.shape)
l = face_recognition.face_locations(test2_photo, model ='cnn')
print(l)
top = l[0][0]
right = l[0][1]
bottom = l[0][2]
left = l[0][3]
test2_image = test2_photo[top:bottom, left:right]
print(test2_image.shape)
test2_face_image = Image.fromarray(test2_image)
test2_face_image.show()


# In[4]:


# Deep Learning based Convolution Neural Network (CNN) approach. CNN
# is more accurate but it takes more time to compute.

from PIL import Image
import face_recognition

# Load the jpg file into a numpy array
group_photo = face_recognition.load_image_file("./pics&group/group1.jpeg")

# Find all the faces in the image 
face_locations = face_recognition.face_locations(group_photo, model="cnn")
# Let us print the number of faces in the Photo
print("There are  {} face(s) in this photo".format(len(face_locations)))
face_count = 0
for face_location in face_locations:
    # Print the location of each face in this image
    face_count = face_count+1
    top, right, bottom, left = face_location
    print("Face {}...Top: {}, Left: {}, Bottom: {}, Right: {}".format(face_count,top, left, bottom, right))
    # You can access the actual face itself like this:
    face_image = group_photo[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()


# In[9]:


group_photo = face_recognition.load_image_file("./pics&group/pic3.jpeg")

# Find all the faces in the image 
face_locations = face_recognition.face_locations(group_photo, model="cnn")
# Let us print the number of faces in the Photo
print("There are  {} face(s) in this photo".format(len(face_locations)))
face_count = 0
for face_location in face_locations:
    # Print the location of each face in this image
    face_count = face_count+1
    top, right, bottom, left = face_location
    print("Face {}...Top: {}, Left: {}, Bottom: {}, Right: {}".format(face_count,top, left, bottom, right))
    # You can access the actual face itself like this:
    face_image = group_photo[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()


#  ## Locate Faces and Mark with rectangle

# In[2]:


from PIL import Image, ImageDraw
import face_recognition
test_photo = face_recognition.load_image_file('./pics&group/pic5.jpeg')
print(test_photo.shape)
l = face_recognition.face_locations(test_photo)
print(l)
top = l[0][0]
right = l[0][1]
bottom = l[0][2]
left = l[0][3]
test_face = test_photo[top:bottom, left:right]
print(test_face.shape)
test_face_image = Image.fromarray(test_face)
#text_face_image.show()
test_photo_image = Image.fromarray(test_photo)
draw = ImageDraw.Draw(test_photo_image)
draw.rectangle(
   (left, top, right, bottom),
   outline = (0, 0, 255), width = 5)
test_photo_image.show()


# ## Writing text on a Face Image

# In[3]:


from PIL import Image, ImageDraw, ImageFont
import face_recognition
photo_text = face_recognition.load_image_file('./pics&group/pic5.jpeg')
print(photo_text.shape)
l = face_recognition.face_locations(photo_text)
print(l)
top = l[0][0]
right = l[0][1]
bottom = l[0][2]
left = l[0][3]


#text_face_image.show()
text_photo_image = Image.fromarray(photo_text)
draw = ImageDraw.Draw(text_photo_image)
font = ImageFont.truetype("arial.ttf", 28, encoding="unic")
draw.rectangle(
   (left, top, right, bottom),
   outline=(0, 0, 255), width = 4)
draw.text((left+100,bottom - 50), "Text on Image", font=font, fill=(255,128,0))
text_photo_image.show()


# ## Locate all faces in a picture

# In[6]:


from PIL import Image, ImageDraw
import face_recognition

# Load the jpg file into a numpy array
group_photo = face_recognition.load_image_file("./pics&group/group1.jpeg")

# Convert the group photo into a PIL Image
group_pil_image = Image.fromarray(group_photo)

# Find all the faces in the image 
fl = face_recognition.face_locations(group_photo, model = 'cnn')
draw = ImageDraw.Draw(group_pil_image)



# Let us print the number of faces in the Photo
face_count = len(fl)
print("No of Faces detected in this photo", face_count)

for i in range(face_count):
    # Print the location of each face in this image
    top, right, bottom, left = fl[i]
    print("Face,,", i, " Top, Left, , Bottom, Right..", top, left, bottom, right)
    # You can access the actual face itself like this:
    draw.rectangle(
            (left, top, right, bottom),
            outline=(255, 0, 0), width=3)
        
group_pil_image.show()


# ## Writing name on all faces

# In[5]:


from PIL import Image, ImageDraw, ImageFont
import face_recognition

# Load the jpg file into a numpy array
grp_photo = face_recognition.load_image_file("./pics&group/group1.jpeg")

# Convert the group photo into a PIL Image
grp_pil_image = Image.fromarray(grp_photo)

# Find all the faces in the image 
fl = face_recognition.face_locations(grp_photo, model ='cnn')
# Let us print the number of faces in the Photo
face_count = len(fl)
print("No of Faces detected in this photo", face_count)

draw = ImageDraw.Draw(grp_pil_image)

fnt = ImageFont.truetype("calibri.ttf", 28, encoding="unic")
# Let us print the number of faces in the Photo

for i in range(face_count):
    # Print the location of each face in this image
    top, right, bottom, left = fl[i]
    print("Face,,", i, " Top, Left, , Bottom, Right..", top, left, bottom, right)
    # You can access the actual face itself like this:
    draw.rectangle(
            (left, top, right, bottom),
            outline=(255, 0, 0), width = 3)
    draw.text((left,bottom-20), str(i), font=fnt, fill=(0,255,0))   
grp_pil_image.show()


# ## Face encoding

# In[9]:


#Program to load a picture and find its face encoding
import face_recognition
photo_emma = face_recognition.load_image_file('./pics&group/pic1.jpeg','RGB')
encodings_emma = face_recognition.face_encodings(photo_emma)[0]
print("The shapes of the Encoding array is ...", encodings_emma.shape)
print("Now let us print the encoding")
print(encodings_emma)


# In[10]:


import face_recognition
photo_gr = face_recognition.load_image_file('./pics&group/group1.jpeg','RGB')
encodings_gr = face_recognition.face_encodings(photo_gr)
print("The Number of faces  ...", len(encodings_gr))
print("The shape of each encodings..", encodings_gr[0].shape)
print("Now let us print the encoding")
for i in range(len(encodings_gr)):
    print(" Encodings of Face...", i)
    print(encodings_gr[i])


# ## Distance Function and Resemblance of Faces
# 
# The purpose of encoding is to find a unique signature for a face. If we consider two face photos of the same person and find out the encodings of each photo. The two encodings are almost same. This means that the Euclidean distance between them is as small as less than 0.6. If the two encodings are for two different person's photos, the distance is more than 0.6. 
# People with resemblance have encodings with distances small and people with totally difference face appearances have encodings of distance high.

# In[11]:


#Program to find distance between faces
import face_recognition
photo_recog1 = face_recognition.load_image_file('./pics&group/pic1.jpeg','RGB')
encodings_recog1 = face_recognition.face_encodings(photo_recog1)[0]

photo_recog2 = face_recognition.load_image_file('./pics&group/pic2.jpeg','RGB')
encodings_recog2 = face_recognition.face_encodings(photo_recog2)[0]
dist = face_recognition.face_distance([encodings_recog1],encodings_recog2)
print("Distance between the two photos ", dist)


# ## Distance between multiple photos

# In[12]:


#Program to load a picture and find its face encoding of same person
import face_recognition
fa = face_recognition.load_image_file('./pics&group/pic1.jpeg','RGB')
f0 = face_recognition.load_image_file('./pics&group/pic2.jpeg','RGB')
f1 = face_recognition.load_image_file('./pics&group/pic3.jpeg','RGB')
f2 = face_recognition.load_image_file('./pics&group/pic4.jpeg','RGB')
f3 = face_recognition.load_image_file('./pics&group/pic5.jpeg','RGB')

fa_sign = face_recognition.face_encodings(fa)[0]
f0_sign = face_recognition.face_encodings(f0)[0]
f1_sign = face_recognition.face_encodings(f1)[0]
f2_sign = face_recognition.face_encodings(f2)[0]
f3_sign = face_recognition.face_encodings(f3)[0]


faces = [f0_sign, f1_sign, f2_sign, f3_sign ]
dist = face_recognition.face_distance(faces,fa_sign)
print("Distance: pic1 to pic2, pic3, pic4,  and pic5 respectively")
print(dist)


# In[13]:


#Program to load five photos of different person
import face_recognition
#Load the five photos of the different person
celebrit1 = face_recognition.load_image_file('./face_recog/celebrit1.jpg','RGB')
celebrit2 = face_recognition.load_image_file('./face_recog/celebrit2.jpg','RGB')
celebrit3 = face_recognition.load_image_file('./face_recog/celebrit3.jpg','RGB')
celebrit4 = face_recognition.load_image_file('./face_recog/celebrit4.jpg','RGB')
celebrit5 = face_recognition.load_image_file('./face_recog/celebrit5.jpg','RGB')

#Let us find the encodings for each of the five faces
encodings_celebrit1 = face_recognition.face_encodings(celebrit1)[0]
encodings_celebrit2 = face_recognition.face_encodings(celebrit2)[0]
encodings_celebrit3 = face_recognition.face_encodings(celebrit3)[0]
encodings_celebrit4 = face_recognition.face_encodings(celebrit4)[0]
encodings_celebrit5 = face_recognition.face_encodings(celebrit5)[0]

# Let us find the distances between faces
dist01_02 = face_recognition.face_distance([encodings_celebrit1],encodings_celebrit2)
dist02_03 = face_recognition.face_distance([encodings_celebrit2],encodings_celebrit3)
dist03_04 = face_recognition.face_distance([encodings_celebrit3],encodings_celebrit4)
dist04_05 = face_recognition.face_distance([encodings_celebrit4],encodings_celebrit5)
dist05_01 = face_recognition.face_distance([encodings_celebrit5],encodings_celebrit1)
print("Distance 1 to 2 ", dist01_02)
print("Distance 2 to 3 ", dist02_03)
print("Distance 3 to 4 ", dist03_04)
print("Distance 4 to 5 ", dist04_05)
print("Distance 5 to 1 ", dist05_01)


# ## Face mapping

# In[14]:


#Father to Son face mapping
# Python program to recognize the photo of a son and map to
# the father depending upon the resemblance.

import face_recognition
n = 6
photo_father =[]
encodings_father = []
photo_son = []
encodings_son = []
for i in range(n):
    f_path_template = './father/father0{}.png'
    f_path = f_path_template.format(i)
    photo_father.append(face_recognition.load_image_file(f_path,'RGB'))
    encodings_father.append(face_recognition.face_encodings(photo_father[i])[0])
    
    s_path_template = './son/son0{}.png'
    s_path = s_path_template.format(i)
    photo_son.append(face_recognition.load_image_file(s_path,'RGB'))
    encodings_son.append(face_recognition.face_encodings(photo_son[i])[0])
    
for i in range(6):
    a = "\nDistance: Son {} to Fathers 0, 1, 2, 3, 4, 5"
    print(a.format(i))
    print(face_recognition.face_distance(encodings_father, encodings_son[i]))


# ## Face Recognition – Compare method
# 
# In this method, compare_ faces() is used to
# compare faces and recognize the faces.
# 
# The input parameters are:
# 1. A list of known face encodings
# 2. One unknown face encoding
# The unknown face is compared with each of the known face fi in the List and a
# Boolean value found(i) is created whether they match or not. The Boolean list found is
# returned.

# In[15]:


import face_recognition
image_emman = face_recognition.load_image_file('./pics&group/pic1.jpeg')
emman_encod = face_recognition.face_encodings(image_emman)[0]

image_em = face_recognition.load_image_file('./pics&group/pic5.jpeg')
em_encod = face_recognition.face_encodings(image_em)[0]
results = face_recognition.compare_faces([emman_encod], em_encod, tolerance = 0.5)
if results[0]:
    print("This is Emmanuel ")
else:
    print('This is NOT Emmanuel ')


# In[16]:


import face_recognition
image_emman = face_recognition.load_image_file('./pics&group/pic1.jpeg')
emman_encod = face_recognition.face_encodings(image_emman)[0]

image_em = face_recognition.load_image_file('./pics&group/pic6.jpeg')
em_encod = face_recognition.face_encodings(image_em)[0]
results = face_recognition.compare_faces([emman_encod], em_encod, tolerance = 0.5)
if results[0]:
    print("This is Emmanuel ")
else:
    print('This is NOT Emmanuel ')


# ## Face compare among unknown pictures

# In[17]:


import face_recognition
pict_1 = face_recognition.load_image_file('./face_recog/celebrit1.jpg','RGB')
pict_2 = face_recognition.load_image_file('./face_recog/celebrit2.jpg','RGB')
pict_3 = face_recognition.load_image_file('./face_recog/celebrit3.jpg','RGB')
pict_4 = face_recognition.load_image_file('./face_recog/celebrit4.jpg','RGB')
pict_5 = face_recognition.load_image_file('./face_recog/celebrit5.jpg','RGB')
pict_6 = face_recognition.load_image_file('./face_recog/pic3.jpeg','RGB')
pict_unkwn = face_recognition.load_image_file('./pics&group/unknown.jpeg','RGB')

pict_unkwn_sign = face_recognition.face_encodings(pict_unkwn)[0]
pict1_sign = face_recognition.face_encodings(pict_1)[0]
pict2_sign = face_recognition.face_encodings(pict_2)[0]
pict3_sign = face_recognition.face_encodings(pict_3)[0]
pict4_sign = face_recognition.face_encodings(pict_4)[0]
pict5_sign = face_recognition.face_encodings(pict_5)[0]
pict6_sign = face_recognition.face_encodings(pict_6)[0]

faces = [pict1_sign, pict2_sign, pict3_sign, pict4_sign, pict5_sign, pict6_sign ]

compare_fnd = face_recognition.compare_faces(faces,pict_unkwn_sign, tolerance = 0.4)

print("Compare Matches:pict1, pict2, pict3, pict4, pict5, pict6")
print(compare_fnd)


# In[18]:


import face_recognition
pict_1 = face_recognition.load_image_file('./face_recog/celebrit1.jpg','RGB')
pict_2 = face_recognition.load_image_file('./face_recog/celebrit2.jpg','RGB')
pict_3 = face_recognition.load_image_file('./face_recog/celebrit3.jpg','RGB')
pict_4 = face_recognition.load_image_file('./face_recog/celebrit4.jpg','RGB')
pict_5 = face_recognition.load_image_file('./face_recog/celebrit5.jpg','RGB')
pict_6 = face_recognition.load_image_file('./face_recog/pic3.jpeg','RGB')
pict_unkwn = face_recognition.load_image_file('./pics&group/pic6.jpeg','RGB')

pict_unkwn_sign = face_recognition.face_encodings(pict_unkwn)[0]
pict1_sign = face_recognition.face_encodings(pict_1)[0]
pict2_sign = face_recognition.face_encodings(pict_2)[0]
pict3_sign = face_recognition.face_encodings(pict_3)[0]
pict4_sign = face_recognition.face_encodings(pict_4)[0]
pict5_sign = face_recognition.face_encodings(pict_5)[0]
pict6_sign = face_recognition.face_encodings(pict_6)[0]

faces = [pict1_sign, pict2_sign, pict3_sign, pict4_sign, pict5_sign, pict6_sign ]

compare_fnd = face_recognition.compare_faces(faces,pict_unkwn_sign, tolerance = 0.4)

print("Compare Matches:pict1, pict2, pict3, pict4, pict5, pict6")
print(compare_fnd)


# In[19]:


import face_recognition
pict_1 = face_recognition.load_image_file('./pics&group/pic1.jpeg','RGB')
pict_2 = face_recognition.load_image_file('./pics&group/pic2.jpeg','RGB')
pict_3 = face_recognition.load_image_file('./pics&group/pic3.jpeg','RGB')
pict_4 = face_recognition.load_image_file('./pics&group/pic4.jpeg','RGB')
pict_5 = face_recognition.load_image_file('./pics&group/pic5.jpeg','RGB')
pict_6 = face_recognition.load_image_file('./pics&group/pic6.jpeg','RGB')
pict_unkwn = face_recognition.load_image_file('./pics&group/unknown.jpeg','RGB')

pict_unkwn_sign = face_recognition.face_encodings(pict_unkwn)[0]
pict1_sign = face_recognition.face_encodings(pict_1)[0]
pict2_sign = face_recognition.face_encodings(pict_2)[0]
pict3_sign = face_recognition.face_encodings(pict_3)[0]
pict4_sign = face_recognition.face_encodings(pict_4)[0]
pict5_sign = face_recognition.face_encodings(pict_5)[0]
pict6_sign = face_recognition.face_encodings(pict_6)[0]

faces = [pict1_sign, pict2_sign, pict3_sign, pict4_sign, pict5_sign, pict6_sign ]

compare_fnd = face_recognition.compare_faces(faces,pict_unkwn_sign, tolerance = 0.4)

print("Compare Matches:pict1, pict2, pict3, pict4, pict5, pict6")
print(compare_fnd)


# ## Image Data storage and compare using Python Pandas CSV File

# In[21]:


import pandas as pd
import face_recognition
f = pd.read_csv('./database/Data.csv')
print(f.to_string())
empno = f["Employee"].tolist()
name = f["Name"].tolist()
filename = f["File Name"].tolist()
n = len(empno)
emp = []
emp_encod = []
ukwn= face_recognition.load_image_file("./pics&group/pic1.jpeg")
ukwn_encod = face_recognition.face_encodings(ukwn)[0]
for i in range(n):
    emp.append(face_recognition.load_image_file(filename[i]))
    emp_encod.append(face_recognition.face_encodings(emp[i])[0])
found = face_recognition.compare_faces(emp_encod, ukwn_encod, tolerance = 0.5)    

print(found)


# In[6]:


import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import face_recognition
fnt = ImageFont.truetype("calibri.ttf", 60, encoding="unic")

f = pd.read_csv('./database/Data.csv')

print(f.to_string())
empno = f["Employee"].tolist()
name = f["Name"].tolist()
filename = f["File Name"].tolist()
n = len(empno)
emp = []
emp_encod = []
ukwn= face_recognition.load_image_file("./pics&group/pic2.jpeg")
print(ukwn.shape)
ukwn_encod = face_recognition.face_encodings(ukwn)[0]
for i in range(n):
    emp.append(face_recognition.load_image_file(filename[i]))
    emp_encod.append(face_recognition.face_encodings(emp[i])[0])
    
found = face_recognition.compare_faces(emp_encod,ukwn_encod, tolerance = 0.4)    
print(found)
for i in range(n):
    if found[i]:
    
        left = 100
        bottom = ukwn.shape[0]
        pil_ukwn = Image.fromarray(ukwn)
        draw = ImageDraw.Draw(pil_ukwn)
        draw.text((left,bottom - 250), name[i], font=fnt, fill=(255,0,0))
        pil_ukwn.show()


# ## Attendance Recording in a File

# In[27]:


import datetime
import pandas as pd
import face_recognition
f = pd.read_csv('./database/Data.csv')
print(f.to_string())
empno = f["Employee"].tolist()
name = f["Name"].tolist()
filename = f["File Name"].tolist()
n = len(empno)
emp = []
emp_encod = []
ukwn= face_recognition.load_image_file("./pics&group/pic3.jpeg")
print(ukwn.shape)
ukwn_encod = face_recognition.face_encodings(ukwn)[0]
for i in range(n):
    emp.append(face_recognition.load_image_file(filename[i]))
    emp_encod.append(face_recognition.face_encodings(emp[i])[0])
    
found = face_recognition.compare_faces(emp_encod,ukwn_encod, tolerance = 0.4)    
print(found)
for i in range(n):
    if found[i]:
        x = str(datetime.datetime.now())
        attend = "\n"+str(empno[i])+' '+str(name[i]+' '+x)
        f = open("Attendance.txt", "a")
        f.write(attend)
        f.close()


# ## Image capture using opencv

# In[28]:


import cv2
camera = cv2.VideoCapture(0)
for i in range(10):
    return_value, image = camera.read()
    print(return_value, image.shape)
    cv2.imwrite('photo'+str(i)+'.png', image)
del(camera)


# In[30]:


import face_recognition
import cv2
from PIL import Image, ImageDraw, ImageFont
import sys
import pandas as pd
import datetime
import pygame

# Module 1 Reference Data Load Module
ef = pd.read_csv('./database/Employee.csv')
empno = ef["Employee No"].tolist()
firstname = ef["First Name"].tolist()
lastname = ef["Last Name"].tolist()
photolocation = ef["Photo Location"].tolist()
audiolocation = ef["Audio Location"].tolist()
n = len(empno)
emp = []
emp_encod = []
audio = []
for i in range(n):
    emp.append(face_recognition.load_image_file(photolocation[i]))
    emp_encod.append(face_recognition.face_encodings(emp[i])[0])


# ## Time and Date Recording

# In[31]:


import datetime
empno = [201, 202, 203, 204, 205, 206, 207,208]
fname = ["mark", "kemi", "nike", "tolu", "tope", "seun"]
lname = ["Anne", "Adesanya","Alao", "Badmus", "Phillip", "Praise"]
x = str(datetime.datetime.now())
i= 5
attendancerecord = "\n"+str(empno[i])+" "+fname[i]+" "+ lname[i]+"  "+x
print(attendancerecord)
f = open("./Attendance.txt", "a")
f.write(attendancerecord)
f.close() 


# ## Playing audio file from the database

# In[32]:


import pygame
pygame.mixer.init()
pygame.mixer.music.load("./database/audio_files/seun.mp3")
pygame.mixer.music.queue("./database/audio_files/Success.mp3")
pygame.mixer.music.play()


# ## Project Work - Employee Attendance Management System
# 
# In this project work I develop an end-to-end attendance management system that uses Face Recognition.
# 
# ## High Level Design
# The following are the key modules:
# 1. Reference Data load Module
# 2. Face Capture and Store temporarily
# 3. Face Recognition
# 4. Attendance Record Module
# 5. Display Attendance Module
# 6. Announce Attendance Module

# In[35]:


import face_recognition
import cv2
from PIL import Image, ImageDraw, ImageFont
import sys
import pandas as pd
import datetime
import pygame

# Module 1 Reference Data Load Module
ef = pd.read_csv('./database/Employee.csv')
empno = ef["Employee No"].tolist()
firstname = ef["First Name"].tolist()
lastname = ef["Last Name"].tolist()
photolocation = ef["Photo Location"].tolist()
audiolocation = ef["Audio Location"].tolist()
n = len(empno)
emp = []
emp_encod = []
audio = []
for i in range(n):
    emp.append(face_recognition.load_image_file(photolocation[i]))
    emp_encod.append(face_recognition.face_encodings(emp[i])[0])


# Module 2 Face Image Capture
camera = cv2.VideoCapture(0)
for i in range(10):
    return_value, image = camera.read()
    cv2.imwrite('Employee'+str(i)+'.png', image)
del(camera)
ukwn =face_recognition.load_image_file('Employee5.png')


#Module 3 Face Recognition Module
def identify_employee(photo):
    try:
        ukwn_encode = face_recognition.face_encodings(photo)[0]
    except IndexError as e:
        print(e)
        sys.exit(1)
    found = face_recognition.compare_faces(
                emp_encod, ukwn_encode, tolerance = 0.4)    
    print(found)
    
    index = -1
    for i in range(n):
        if found[i]:
            index = i
    return(index)

emp_index = identify_employee(ukwn)    
print(emp_index)   


# Module 4 Attendance record in a data file attendance.txt
if (emp_index != -1):
    name ="Face NOT Recognized"
    x = str(datetime.datetime.now())
    eno = str(empno[emp_index])
    f = firstname[emp_index]
    l = lastname[emp_index]
    ar = "\n"+eno+" "+f+" "+ l+ "  "+x
    f = open("./Attendance.txt", "a")
    f.write(ar)
    f.close()  
    print(ar)
    
    
# Module 5 Display Attendance Module
pil_ukwn = Image.fromarray(ukwn)
draw = ImageDraw.Draw(pil_ukwn)
fnt = ImageFont.truetype("calibri.ttf", 60, encoding="unic")

if emp_index ==-1:
    name ="Face NOT Recognized"
else:
    name = firstname[emp_index]+" "+lastname[emp_index]
x = 100
y = ukwn.shape[0] - 100
draw.text((x, y), name, font=fnt, fill=(255,0,0))
pil_ukwn.show()


# Module 6 Announce Attendance Recorded Module
audioloc = audiolocation[emp_index]
pygame.mixer.init()
if emp_index ==-1:
    pygame.mixer.music.load("./database/audio_files/not_recog.mp3")
    pygame.mixer.music.play()
else:
    pygame.mixer.music.load(audioloc)
    pygame.mixer.music.play()
    pygame.mixer.music.queue("./database/audio_files/success.mp3")
    pygame.mixer.music.play()
    
