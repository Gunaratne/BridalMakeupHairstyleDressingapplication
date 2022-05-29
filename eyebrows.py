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
        imgRightEyebrows = createBox(img, myPoints[23:27], 20, masked=True, cropped=False)
        imgLeftEyebrows = createBox(img, myPoints[18:22], 20, masked=True, cropped=False)

        imgColorRightEyebrows = np.zeros_like(imgRightEyebrows)
        imgColorLeftEyebrows = np.zeros_like(imgLeftEyebrows)
        b = cv2.getTrackbarPos('Blue', 'BGR')
        g = cv2.getTrackbarPos('Green', 'BGR')
        r = cv2.getTrackbarPos('Red', 'BGR')
        imgColorRightEyebrows[:] = b, g, r
        imgColorLeftEyebrows[:] = b, g, r
        imgColorRightEyebrows = cv2.bitwise_and(imgRightEyebrows, imgColorRightEyebrows)
        imgColorLeftEyebrows = cv2.bitwise_and(imgLeftEyebrows, imgColorLeftEyebrows)
        imgColorRightEyebrows = cv2.GaussianBlur(imgColorRightEyebrows, (7, 7), 10)
        imgColorLeftEyebrows = cv2.GaussianBlur(imgColorLeftEyebrows, (7, 7), 10)
        imgColorRightEyebrows = cv2.addWeighted(img, 1, imgColorRightEyebrows, 0.4, 0)
        imgColorLeftEyebrows = cv2.addWeighted(img, 1, imgColorLeftEyebrows, 0.4, 0)
        finalview = cv2.add(imgColorLeftEyebrows, imgColorRightEyebrows)
        cv2.imshow("BGR", finalview)

    cv2.waitKey(1)