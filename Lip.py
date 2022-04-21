#import the libs
import cv2
import numpy as np
import dlib

webcam = True
cap =cv2.VideoCapture(0)

dtector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#Color change scale
def empty(a):
    pass

cv2.namedWindow("BGR")
cv2.resizeWindow("BGR",640,240)
cv2.createTrackbar("Blue",'BGR',0,255,empty)
cv2.createTrackbar("Green",'BGR',0,255,empty)
cv2.createTrackbar("Red",'BGR',0,255,empty)

def createBox(img,points,scale=5,masked=False,cropped=True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [points], (255, 255, 255))
        img = cv2.bitwise_and(img, mask)

    if cropped:
        bbox = cv2.boundingRect(points)
        x, y, w, h = bbox
        imgCrop = img[y:y + h, x:x + w]
        imgCrop = cv2.resize(imgCrop, (0, 0), None, scale, scale)
        return imgCrop

    else:
        return mask

while True:

    if webcam: success, img = cap.read()
    else:img = cv2.VideoCapture(0)
    img = cv2.resize(img, (0, 0), None, 1, 1)
    faces = dtector(img)

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        landmarks = predictor(img, face)
        myPoints = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x, y])

        myPoints = np.array(myPoints)
        imgLip = createBox(img, myPoints[48:61], 6, masked=True, cropped=False)

        imgColorLip = np.zeros_like(imgLip)
        b = cv2.getTrackbarPos('Blue', 'BGR')
        g = cv2.getTrackbarPos('Green', 'BGR')
        r = cv2.getTrackbarPos('Red', 'BGR')
        imgColorLip[:] = b, g, r
        imgColorLip = cv2.bitwise_and(imgLip, imgColorLip)
        imgColorLip = cv2.GaussianBlur(imgColorLip, (7, 7), 10)
        imgColorLip = cv2.addWeighted(img, 1, imgColorLip, 0.4, 0)
        cv2.imshow('BGR', imgColorLip)

    cv2.waitKey(1)