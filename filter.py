import cv2
import numpy as np

# Load video
# cap = cv2.VideoCapture('sample360.mp4')
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()

    if not ret:
        break

    # Convert to grayscale
       
    gray = np.float64(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))/255
    
    lp = cv2.GaussianBlur(gray, (5,5), 0)
    lp2 = cv2.GaussianBlur(gray, (23,23), 0)
    hp = gray - lp
    hp2 = gray - lp2
    br = lp + hp2
    bp = gray - br

    # Display the frames
    cv2.imshow('Default', gray)
    cv2.imshow('Lowpass Filter', lp)
    cv2.imshow('Lowpass2 Filter', lp2)
    cv2.imshow('Highpass Filter', hp*20)
    cv2.imshow('Highpass2 Filter', hp2*20)
    cv2.imshow('Band Rejected Filter', br*1.5)
    cv2.imshow('Bandpass Filter', bp*20)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

cap.release()
cv2.destroyAllWindows()




# def lowpass_filter(img, kernel_size):
#     kernel = np.ones(kernel_size, np.float32) / (kernel_size[0] * kernel_size[1])
#     return cv2.filter2D(img, -1, kernel)

# def highpass_filter(img, kernel_size):
#     return cv2.subtract(img, lowpass_filter(img, kernel_size))

# def band_rejected_filter(img, kernel_size_low, kernel_size_high):
#     lowpass_result = lowpass_filter(img, kernel_size_low)
#     highpass_result = highpass_filter(img, kernel_size_high)
#     return cv2.subtract(highpass_result, lowpass_result)

# def bandpass_filter(img, kernel_size_low, kernel_size_high):
#     return cv2.subtract(img, band_rejected_filter(img, kernel_size_low, kernel_size_high))

# Apply filters
# lowpass_result = lowpass_filter(gray, (3, 7)) 
# highpass_result = highpass_filter(gray, (3, 7)) *20
# band_rejected_result = lowpass_result - highpass_filter(gray, (5,7))
# bandpass_result = bandpass_filter(gray, (3, 7), (5, 7)) * 10