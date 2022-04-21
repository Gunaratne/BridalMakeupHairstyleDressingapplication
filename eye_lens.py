#import the libs
import cv2
import numpy as np
import dlib


webcam = True
cap = cv2.VideoCapture(0)

####prepeare the dlib
detector = dlib.get_frontal_face_detector()
#add predictor
predictor = dlib.shape_predictor("shape_predictor_70_face_landmarks.dat")

#Color change scale
def empty(a):
    pass

cv2.namedWindow("BGR")
cv2.resizeWindow("BGR",640,240)
cv2.createTrackbar("Blue",'BGR',0,255,empty)
cv2.createTrackbar("Green",'BGR',0,255,empty)
cv2.createTrackbar("Red",'BGR',0,255,empty)

while True:

    if webcam: success, img = cap.read()
    else:
        #######image prepeeration
        # loaf the image
        img = cv2.VideoCapture(0)

    # start to dect the face in the image
    faces = detector(img)

    # center of eye
    landmarkpts = []

    # many face detector
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        # predict the face component
        landmarks = predictor(img, face)
        for n in range(68, 70):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # 4 eyes
            landmarkpts.append([x, y])

        # preparing the mask
        mask = np.zeros_like(img)
        # two circal for mask
        lefteye = cv2.circle(mask, (landmarkpts[0][0], landmarkpts[0][1]), 6, (255, 255, 255), cv2.FILLED)
        rightteye = cv2.circle(mask, (landmarkpts[1][0], landmarkpts[1][1]), 6, (255, 255, 255), cv2.FILLED)

        # color
        b, r, g = 23, 222, 1
        eyecolr = np.zeros_like(mask)
        b = cv2.getTrackbarPos('Blue', 'BGR')
        g = cv2.getTrackbarPos('Green', 'BGR')
        r = cv2.getTrackbarPos('Red', 'BGR')
        eyecolr[:] = b, g, r
        eyecolormask = cv2.bitwise_and(mask, eyecolr)
        # add blead
        eyecolormask = cv2.GaussianBlur(eyecolormask, (7, 7), 10)
        # add eye color mask to image
        finaling = cv2.addWeighted(img, 1, eyecolormask, 0.4, 0)
        cv2.imshow("BGR", finaling)

    cv2.waitKey(1)