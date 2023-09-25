import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cutoff_low = 30
cutoff_high = 50

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    
    ret, frame = cap.read()
    gray = np.float64(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))/255

     # Robert
    gx = np.array([[-1,1]])
    gy = np.array([[-1,1]]).transpose()

    imx = (cv2.filter2D(gray, cv2.CV_64F, gx))*15
    imy = (cv2.filter2D(gray, cv2.CV_64F, gy))*15

    #Magnitude Manual
    grad = np.sqrt((imx * imx) + (imy * imy))
    #Magnitude OpenCV
    # grad_cv = cv2.magnitude(imx,imy)
   
    cv2.imshow('frame', frame)
    cv2.imshow('gradien x', imx)
    cv2.imshow('gradien y', imy)
    cv2.imshow("gradien total", grad)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        break   

cap.release
cv2.destroyAllWindows()


 # # HP LP
    # lp = cv2.GaussianBlur(gray, (3,3), cutoff_low)
    # hp = gray - lp

   

    # # Band Reject
    # br = hp - cv2.GaussianBlur(hp, (3,3), cutoff_high)

    # # Bandpass
    # bp = gray - br