import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Import images
path = 'ImageAttendance'

# create list for all images to be imported
Images = []
ClassNames = []
myList = os.listdir(path)

# import images one by one
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    Images.append(curImg)
    ClassNames.append(os.path.splitext(cl)[0])
print(ClassNames)

# create a function to compute encoding
def findEncodings(Images):
    encodeList = []

# Loop through all the images
    for Img in Images:
        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)

# Find the encodings
        encode = face_recognition.face_encodings(Img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(Images)
print('Encoding complete')

# Find the matches between encodings using webcam
cap = cv2.VideoCapture(0)

while True:
    success, Img = cap.read()

# Reduce images size to speed the process since this is done in real time
    ImgS = cv2.resize(Img,(0,0),None,0.25,0.25)
    ImgS = cv2.cvtColor(ImgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(ImgS)
    encodeCurFrame = face_recognition.face_encodings(ImgS,facesCurFrame)

# Find the matches
    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

        # print(faceDis)
        matchIndex = np.argmin(faceDis)

# Display bounding box and display name
        if matches[matchIndex]:
            name = ClassNames[matchIndex].upper()
            print(name)

            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(Img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(Img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(Img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

        cv2.imshow('webcam',Img)
        cv2.waitKey(1)