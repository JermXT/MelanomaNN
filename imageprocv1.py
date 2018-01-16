#cat img > input.txt | python pathy.py

import numpy as np
import cv2, sys, math, codecs, time

#closing = np.ones((4,4),np.uint8)
kernel = np.ones((2,2),np.uint8)
ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
np.array([[0, 0, 1, 0, 0],
          [0, 1, 1, 1, 0],
          [1, 1, 1, 1, 1],
          [0, 1, 1, 1, 0],
          [0, 0, 1, 0, 0]], dtype=np.uint8)

"""
np.array([[1, 1, 0, 0, 0],
       [1, 1, 1, 0, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 1, 1, 1],
       [0, 0, 0, 1, 1]], dtype=np.uint8)
"""

img = cv2.imread("Lesion.png")
cv2.imshow("original",img)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, ellipse)
#reclo = cv2.morphologyEx(img, cv2.MORPH_CLOSE, closingt1)
bb = abs(img[:,:,0]-closing[:,:,0])
gg = abs(img[:,:,1]-closing[:,:,1])
rr = abs(img[:,:,2]-closing[:,:,2])
slices = np.bitwise_and(bb,np.bitwise_and(gg, rr, dtype=np.uint8), dtype = np.uint8)*255
slices = cv2.dilate(slices,kernel,iterations = 2)
"""
closing = cv2.cvtColor(closing,cv2.COLOR_BGR2LAB)
imgLAB = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
new = cv2.subtract(imgLAB, closing)
#new = cv2.cvtColor(new, cv2.COLOR_LAB2BGR)
"""
for y in range(len(slices)):
        for x in range(len(slices[0])):
        	if(slices[y][x] > 20):
                	img[y][x] = closing[y][x]


cv2.imshow("image", img)
#cv2.imshow("nohair-re", reclo)
cv2.imshow("nohair-el", closing)
cv2.imshow("thresh", slices)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
var = img[:,:,0] < 20;

slices = np.bitwise_and(br,np.bitwise_and(gr, rr, dtype=uint8), dtype = uint8)*255
"""
