import numpy as np
import cv2, sys, math, codecs, time
#from matplotlib import pyplot as plt

kernel = np.ones((2,2),np.uint8)
# kernel to dialate thresh

ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
np.array([[0, 0, 1, 0, 0],
          [0, 1, 1, 1, 0],
          [1, 1, 1, 1, 1],
          [0, 1, 1, 1, 0],
          [0, 0, 1, 0, 0]], dtype=np.uint8)
# kernel for closing


img = cv2.imread("LesionEdit.png")

cv2.imshow("original",img)

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, ellipse)

bb = abs(img[:,:,0]-closing[:,:,0])
gg = abs(img[:,:,1]-closing[:,:,1])
rr = abs(img[:,:,2]-closing[:,:,2])
# original vs closing

slices = np.bitwise_and(bb,np.bitwise_and(gg, rr, dtype=np.uint8), dtype = np.uint8)*255
slices = cv2.dilate(slices,kernel,iterations = 2)
# combines channels and then dialates them

for y in range(len(slices)):
        for x in range(len(slices[0])):
        	if(slices[y][x] > 20):
                	img[y][x] = closing[y][x]
# replaces hair pixels with closing image pixels
cv2.imshow("nohair",img)
cv2.imshow("hair", slices)
# laplacian edge detection, shows random crap
#laplacian64 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#laplacian64 = cv2.Laplacian(laplacian64, cv2.CV_64F)
#laplacian = np.uint8(np.absolute(laplacian64))
#laplacian = cv2.Laplacian(img,cv2.CV_64F)
#sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
#edges = cv2.Canny(img,50,200)


"""
# simple blob, not useful
#detector = cv2.SimpleBlobDetector()
#keypoints = detector.detect(img)

params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200

# Filter by Area.
params.filterByArea = True
params.minArea = 1500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(img)

im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
"""

canvas = np.zeros(img.shape, np.uint8)
img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# filter out small lines between counties
#kernel = np.ones((5,5),np.float32)/25
#img2gray = cv2.filter2D(img2gray,-1,kernel)

# threshold the image and extract contours
ret,thresh = cv2.threshold(img2gray,250,255,cv2.THRESH_BINARY_INV)
im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


# find the main island (biggest area)
cnt = contours[0]
max_area = cv2.contourArea(cnt)

for cont in contours:
    if cv2.contourArea(cont) > max_area:
        cnt = cont
        max_area = cv2.contourArea(cont)

# define main island contour approx. and hull
perimeter = cv2.arcLength(cnt,True)
epsilon = 0.01*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

hull = cv2.convexHull(cnt)

# cv2.isContourConvex(cnt)

cv2.drawContours(canvas, cnt, -1, (0, 255, 0), 3)
cv2.drawContours(canvas, approx, -1, (0, 0, 255), 3)
## cv2.drawContours(canvas, hull, -1, (0, 0, 255), 3) # only displays a few points as well.

cv2.imshow("Contour", canvas)






cv2.imshow("image", img)
#cv2.imshow("nohair-el", closing)
#cv2.imshow("thresh", slices)
cv2.waitKey(0)
cv2.destroyAllWindows()


