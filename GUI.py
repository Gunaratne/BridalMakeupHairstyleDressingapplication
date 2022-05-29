#import the libs
import cv2
import numpy as np
import dlib

webcam = True
cap =cv2.VideoCapture(0)

dtector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
predictor1 = dlib.shape_predictor("shape_predictor_70_face_landmarks.dat")



#Color change scale
def empty(a):
    pass

cv2.namedWindow("BGR")
cv2.resizeWindow("BGR",640,240)
cv2.createTrackbar("Lip Blue",'BGR',0,255,empty)
cv2.createTrackbar("Lip Green",'BGR',0,255,empty)
cv2.createTrackbar("Lip Red",'BGR',0,255,empty)
cv2.createTrackbar("Len Blue",'BGR',0,255,empty)
cv2.createTrackbar("Len Green",'BGR',0,255,empty)
cv2.createTrackbar("Len Red",'BGR',0,255,empty)
cv2.createTrackbar("Eyebrows Blue",'BGR',0,255,empty)
cv2.createTrackbar("Eyebrows Green",'BGR',0,255,empty)
cv2.createTrackbar("Eyebrows Red",'BGR',0,255,empty)

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
    landmarkpts = []

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        landmarks = predictor(img, face)
        landmarks1 = predictor1(img, face)
        myPoints = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x, y])

        for n in range(68, 70):
            x = landmarks1.part(n).x
            y = landmarks1.part(n).y
            # 4 eyes
            landmarkpts.append([x, y])

        myPoints = np.array(myPoints)
        imgLip = createBox(img, myPoints[48:61], 6, masked=True, cropped=False)
        imgRightEyebrows = createBox(img, myPoints[23:27], 30, masked=True, cropped=False)
        imgLeftEyebrows = createBox(img, myPoints[18:22], 30, masked=True, cropped=False)

        mask = np.zeros_like(img)
        # two circal for mask
        lefteye = cv2.circle(mask, (landmarkpts[0][0], landmarkpts[0][1]), 10, (255, 255, 255), cv2.FILLED)
        rightteye = cv2.circle(mask, (landmarkpts[1][0], landmarkpts[1][1]), 10, (255, 255, 255), cv2.FILLED)

        imgColorLip = np.zeros_like(imgLip)
        imgColorRightEyebrows = np.zeros_like(imgRightEyebrows)
        imgColorLeftEyebrows = np.zeros_like(imgLeftEyebrows)
        #b1, r1, g1 = 23, 222, 1
        eyecolr = np.zeros_like(mask)
        b = cv2.getTrackbarPos('Lip Blue', 'BGR')
        g = cv2.getTrackbarPos('Lip Green', 'BGR')
        r = cv2.getTrackbarPos('Lip Red', 'BGR')
        b1 = cv2.getTrackbarPos('Len Blue', 'BGR')
        g1 = cv2.getTrackbarPos('Len Green', 'BGR')
        r1 = cv2.getTrackbarPos('Len Red', 'BGR')
        b2 = cv2.getTrackbarPos('Eyebrows Blue', 'BGR')
        g2 = cv2.getTrackbarPos('Eyebrows Green', 'BGR')
        r2 = cv2.getTrackbarPos('Eyebrows Red', 'BGR')

        imgColorLip[:] = b, g, r
        eyecolr[:] = b1, g1, r1
        imgColorRightEyebrows[:] = b2, g2, r2
        imgColorLeftEyebrows[:] = b2, g2, r2
        imgColorLip = cv2.bitwise_and(imgLip, imgColorLip)
        imgColorRightEyebrows = cv2.bitwise_and(imgRightEyebrows, imgColorRightEyebrows)
        imgColorLeftEyebrows = cv2.bitwise_and(imgLeftEyebrows, imgColorLeftEyebrows)
        imgColorLip = cv2.GaussianBlur(imgColorLip, (7, 7), 10)
        imgColorRightEyebrows = cv2.GaussianBlur(imgColorRightEyebrows, (7, 7), 10)
        imgColorLeftEyebrows = cv2.GaussianBlur(imgColorLeftEyebrows, (7, 7), 10)
        imgColorLip = cv2.addWeighted(img, 1, imgColorLip, 0.4, 0)
        imgColorRightEyebrows = cv2.addWeighted(img, 1, imgColorRightEyebrows, 0.4, 0)
        imgColorLeftEyebrows = cv2.addWeighted(img, 1, imgColorLeftEyebrows, 0.4, 0)
        #cv2.imshow('BGR', imgColorLip)
        eyecolormask = cv2.bitwise_and(mask, eyecolr)
        # add blead
        eyecolormask = cv2.GaussianBlur(eyecolormask, (7, 7), 10)
        # add eye color mask to image
        finaling = cv2.addWeighted(img, 1, eyecolormask, 0.4, 0)
        Eyebrows = cv2.add(imgColorLeftEyebrows,imgColorRightEyebrows)
        LipLen = cv2.add(imgColorLip,finaling)
        finalview = cv2.add(Eyebrows,LipLen)
        cv2.imshow("BGR", imgColorLeftEyebrows)

    cv2.waitKey(1)