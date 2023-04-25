import cv2
import numpy as np

image = cv2.imread('blueflower.jpg')
#Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# define blue color range
light_blue = np.array([110,50,50])
dark_blue = np.array([130,255,255])
#Threshold the HSV image to get only blue color
mask = cv2.inRange(hsv, light_blue, dark_blue)
# Bitwise-AND mask and orginal image
output = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Color Detected", np.hstack((image, output)))
cv2.waitKey(0)
cv2.destroyAllWidows()