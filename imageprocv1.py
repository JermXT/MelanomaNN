import numpy as np
import cv2, sys, math, codecs, time
#from matplotlib import pyplot as plt

kernel = np.ones((2,2),np.uint8)
# kernel to dialate thresh
r=0
g=0
b=0

ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
np.array([[0, 0, 1, 0, 0],
          [0, 1, 1, 1, 0],
          [1, 1, 1, 1, 1],
          [0, 1, 1, 1, 0],
          [0, 0, 1, 0, 0]], dtype=np.uint8)
# kernel for closing


img = cv2.imread("detailed.png")


cv2.imshow("original",img)

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, ellipse)

bb = abs(img[:,:,0]-closing[:,:,0])
gg = abs(img[:,:,1]-closing[:,:,1])
rr = abs(img[:,:,2]-closing[:,:,2])
# original vs closing

slices = np.bitwise_and(bb,np.bitwise_and(gg, rr, dtype=np.uint8), dtype = np.uint8)*255
slices = cv2.dilate(slices,kernel,iterations = 2)
# combines channels and then dialates them

groups = []
for y in range(len(slices)):
        for x in range(len(slices[0])):
        	if(slices[y][x] > 20):
                	img[y][x] = closing[y][x]
# replaces hair pixels with closing image pixels

for y in range(len(img)):
	for x in range(len(img[0])):
		b+=img[y][x][0]
		g+=img[y][x][1]
		r+=img[y][x][2]
r = r/(len(img)*len(img[0]))*0.8
g = g/(len(img)*len(img[0]))*0.8
b = b/(len(img)*len(img[0]))*0.8

"""
for y in range(len(img)):
        for x in range(len(img[0])):
		if(img[y][x][0] < b and img[y][x][1] < g and img[y][x][2] < r):
			img[y][x][0] =0
			img[y][x][1] =0
			img[y][x][2] =0
"""
#cv2.imshow("imageedit", img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
maxcoordx = 0
currentx = 255
currenty = 255
maxcoordy =0
for y in range(len(img)):
	average = 0;
	for x in range(len(img[0])):
		average += img[y][x]
	#print average
	average=float(average)/float(len(img[0]))
	#print average
	if average < currenty:
		currenty = average
		maxcoordy = y

for x in range(len(img[0])):
        average = 0;
        for y in range(len(img)):
                average += img[y][x]
        average=float(average)/float(len(img[0]))
        if average < currentx:
                currentx = average
		maxcoordx = x
print(maxcoordx,maxcoordy)

cv2.circle(img,(maxcoordx,maxcoordy),10,(0,0,255),-1)
cv2.imshow("center",img)

#cv2.imshow("nohair-el", closing)
#cv2.imshow("thresh", slices)
cv2.waitKey(0)
cv2.destroyAllWindows()


