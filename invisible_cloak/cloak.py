import cv2
import numpy as np
import time
import argparse

# Creating an VideoCapture object
# This will be used for image acquisition later in the code.
cap = cv2.VideoCapture('Input.mp4')

# We give some time for the camera to setup
time.sleep(3)
count = 0
background = 0

ret = None
background = None

# Capturing and storing the static background frame
for i in range(60):
    ret, background = cap.read()

#background = np.flip(background,axis=1)

outputHeight = 0
outputWidth = 0

# Get the original video res
if cap.isOpened():
    outputWidth = int(cap.get(3)) 
    outputHeight = int(cap.get(4))  

# Create the output video stream
out = cv2.VideoWriter(
    'output.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (outputWidth, outputHeight))

while(cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    count += 1
    # Converting the color space from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Generating mask to detect red color
    lower_range = np.array([22, 35, 17])
    upper_range = np.array([69, 255, 255])
    mask1 = cv2.inRange(hsv, lower_range, upper_range)
    #kernel_gg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    # Create the kernel element for refinement
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))

    # Apply an opening and a closing operation
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel, iterations=2)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Create the kernel element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # Apply an opening and a closing operation
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel, iterations=2)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Apply a small dilation to ensure that the edges are fully covered
    mask1 = cv2.dilate(mask1, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (19, 19)), iterations=1)

    # Obtain the inverted mask
    mask2 = cv2.bitwise_not(mask1)

    # Extract the two parts from the background and the current frame
    im1 = cv2.bitwise_and(background, background, mask=mask1)
    im2 = cv2.bitwise_and(img, img, mask=mask2)

    # Combine the two results together
    final_output = cv2.addWeighted(im1, 1, im2, 1, 0)
    print(final_output.shape, count)
    out.write(final_output)

print('Releasing')
out.release()

