import cv2
import imutils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

image = cv2.imread('image1.png', cv2.IMREAD_UNCHANGED)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:

    ret, frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame, width=640)

    faces = faceClassif.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:

        pic_resized = cv2.resize(image, (2888, 5000))
        resized_image = imutils.resize(pic_resized, width=w)

        filas_image = resized_image.shape[0]
        col_image = w


        porcion_alto = filas_image // 2

        dif = 0


        if y + porcion_alto - filas_image >= 0:


            n_frame = frame[y + porcion_alto - filas_image: y + porcion_alto,
                      x: x + col_image]
        else:

            dif = abs(y + porcion_alto - filas_image)
            n_frame = frame[0: y + porcion_alto,
                      x: x + col_image]


        mask = resized_image[:, :, 3]
        mask_inv = cv2.bitwise_not(mask)


        bg_black = cv2.bitwise_and(resized_image, resized_image, mask=mask)
        bg_black = bg_black[dif:, :, 0:3]
        bg_frame = cv2.bitwise_and(n_frame, n_frame, mask=mask_inv[dif:, :])


        result = cv2.add(bg_black, bg_frame)
        if y + porcion_alto - filas_image >= 0:
            frame[y + porcion_alto - filas_image: y + porcion_alto, x: x + col_image] = result

        else:
            frame[0: y + porcion_alto, x: x + col_image] = result

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()