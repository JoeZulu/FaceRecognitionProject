import cv2

frameWidth = 640
frameHeight = 360

cap = cv2.VideoCapture("Resources/"
                       "Sorry.mp4")

while True:
    success,img = cap.read()
    cv2.imshow("video",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break