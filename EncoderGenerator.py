import os
import cv2
import face_recognition
import pickle

forlderPath = 'images'
pathList = os.listdir(forlderPath)
print(pathList)
imgList = []
studentIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(forlderPath, path)))
    studentIds.append(os.path.splitext(path)[0])


def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


print('Encoding...')
encodeListKnown = findEncodings(imgList)
encodeListKnownIds = [encodeListKnown, studentIds]
print (encodeListKnownIds)
print('Encoding Complete')

file = open('EncodeFile.p', 'wb')
pickle.dump(encodeListKnownIds, file)
file.close()
print('File Created')
