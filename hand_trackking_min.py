import cv2
import mediapipe as mp
import time 


cap = cv2.VideoCapture(0)



mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpdraw =mp.solutions.drawing_untils


while True:
    succsess, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results  = hands.process(imgRGB)
    # print (results.multi_hand._landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_landmarks:
            mpdraw.draw_landmarks(img,handLms)

    cv2.imshow("image", img)
    cv2.waitKey(1)


