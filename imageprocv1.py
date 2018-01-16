import numpy as np
import cv2, sys, math, codecs, time


kernel = np.ones((2,2),np.uint8)
# kernel to dialate thresh

ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
np.array([[0, 0, 1, 0, 0],
          [0, 1, 1, 1, 0],
          [1, 1, 1, 1, 1],
          [0, 1, 1, 1, 0],
          [0, 0, 1, 0, 0]], dtype=np.uint8)
# kernel for closing


img = cv2.imread("Lesion1.png")

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

cv2.imshow("image", img)
cv2.imshow("nohair-el", closing)
cv2.imshow("thresh", slices)
cv2.waitKey(0)
cv2.destroyAllWindows()


