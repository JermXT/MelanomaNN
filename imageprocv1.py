import random
import numpy as np
import cv2, sys, math, codecs, time, copy

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


img = cv2.imread("Images/detailed.png")
#img = cv2.imread("Images/ISIC_0000016.png")
#cv2.imshow("original",img)

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
pix = len(img)*len(img[0])
r = r/(len(img)*len(img[0])*1.0)
g = g/(len(img)*len(img[0])*1.0)
b = b/(len(img)*len(img[0])*1.0)
print(b,g,r)
size = 10
if(pix < 240 * 240/9*16):
	size = 4
elif(pix < 480 * 480/9*16):
	size = 6
elif(pix < 720 * 720/9*16):
	size = 8
size = 10

imgbw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
maxcoordx = 0
currentx = 255
currenty = 255
maxcoordy =0
for y in range(len(img)):
	average = 0;
	for x in range(len(img[0])):
		average += imgbw[y][x]
		#print average
		average=float(average)/float(len(img[0]))
		#print average
		if average < currenty:
			currenty = average
			maxcoordy = y

for x in range(len(img[0])):
        average = 0;
        for y in range(len(img)):
                average += imgbw[y][x]
        average=float(average)/float(len(img[0]))
        if average < currentx:
                currentx = average
		maxcoordx = x
print(maxcoordx,maxcoordy)
run = True
current = maxcoordx
newImg=copy.deepcopy(img)
while(True):
	if(img[maxcoordy][current][0]<=b and img[maxcoordy][current][1]<=g and img[maxcoordy][current][2]<=r):
		newImg[maxcoordy][current][0] =0
	elif(img[maxcoordy+1][current][0]<=b and img[maxcoordy+1][current][1]<=g and img[maxcoordy+1][current][2]<=r):
                newImg[maxcoordy+1][current][0] =0
	elif(img[maxcoordy-1][current][0]<=b and img[maxcoordy-1][current][1]<=g and img[maxcoordy-1][current][2]<=r):
                newImg[maxcoordy-1][current][0] =0
	else:
		break
	#print((img[maxcoordy][current][0],img[maxcoordy][current][1],img[maxcoordy][current][2]))
	#if current ==0 :
	#	break
	current = current - 1
i = 10000
prev = [1,0]
coords = [current+1, maxcoordy]
print coords
#[[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]]
order = [[-1,0],[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1]]
#visited1 =[coords]
#visited2 = []
while(i>=0):
	index = order.index(prev)
	for x in range((index+1), (index+9)):
		#print order[x]
		point = img[coords[1]+order[x%8][1]][coords[0]+order[x%8][0]]
		#print point
		#tobreak = False
		#coords = [coords[0]+order[x%8][0],coords[1]+order[x%8][1]]
		if(point[0]<=b and point[1]<=g and point[2]<=r):
			#if(False==(coords in visited1)):
                             #print "hi"
                             #print "this:", point
			     newImg[coords[1]+order[x%8][1]][coords[0]+order[x%8][0]][1] = 0
			     prev[0] = order[x%8][0]
			     prev[1] = order[x%8][1]
			     prev[0]=prev[0]*-1
			     prev[1]=prev[1]*-1
			     #print order[x%8]
			     coords = [coords[0]+order[x%8][0],coords[1]+order[x%8][1]]
			     #print coords
			     #if(coords in visited1):
                             #    visited2.append(coords)
                             #else:
                             #   visited1.append(coords)
			     #visited1.append(coords)
			     break
                             #print point
		
	i = i-1
#print coords
#below finds center spot via checking of dark sq
"""
satisfied = False
#for x in range(10):
#	print(random.randrange(len(img[0])/4,len(img[0])*3/4,1),random.randrange(len(img)/4,len(img)*3/4,1))
def findspot(satisfied):
	if(satisfied == False):
		cont = True
		#x = random.randrange(len(img[0])/4,len(img[0])*3/4,1)
		#y = random.randrange(len(img)/4,len(img)*3/4,1)
		x = 700
		y=440
		print(x,y)
		for yc in range(y, y+size):
			for xc in range(x, x+size):
				if(img[yc][xc][0] > b or img[yc][xc][1] > g or img[yc][xc][2] > r):
					cont = False 
					array = findspot(False)
		if cont == True:
			cv2.rectangle(img, (x,y),(x+size,y+size),(255,255,0),3)
			return [y,x]
ar = findspot(satisfied)
"""

"""
for y in range(len(img)):
        for x in range(len(img[0])):
		if(img[y][x][0] < b and img[y][x][1] < g and img[y][x][2] < r):
			img[y][x][0] =0
			img[y][x][1] =0
			img[y][x][2] =0
"""

cv2.imshow("imageedit", newImg)

#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#maxcoordx = 0
#currentx = 255
#currenty = 255
#maxcoordy =0
#for y in range(len(img)):
#	average = 0;
#	for x in range(len(img[0])):
		
#cv2.imshow("nohair-el", closing)
#cv2.imshow("thresh", slices)
cv2.waitKey(0)
cv2.destroyAllWindows()


