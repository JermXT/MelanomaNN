#import random
import numpy as np
import cv2, math, copy, imutils, os, datetime

#from matplotlib import pyplot as plt
def run(filename,const):
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

	img = cv2.imread(filename)
	#img = cv2.imread("Images/detailed.png")
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
	r = r/(len(img)*len(img[0])*const)
	g = g/(len(img)*len(img[0])*const)
	b = b/(len(img)*len(img[0])*const)
	#print(b,g,r)
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
	for y in range(5,len(img)-5):
		average = 0;
		for x in range(5,len(img[0])-5):
			average += imgbw[y][x]
			#print average
			average=float(average)/float(len(img[0]))
			#print average
			if average < currenty:
				currenty = average
				maxcoordy = y
	
	for x in range(5,len(img[0])-5):
        	average = 0;
        	for y in range(5,len(img)-5):
        	        average += imgbw[y][x]
        	average=float(average)/float(len(img[0]))
        	if average < currentx:
                	currentx = average
			maxcoordx = x
	#print(maxcoordx,maxcoordy)
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
	i = 1000000
	prev = [1,0]
	coords = [current, maxcoordy]
	#print coords
	#[[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]]
	order = [[-1,0],[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1]]
	#visited1 =[coords]
	#visited2 = []
	contour =[[current, maxcoordy]]
	approx = 15
	exit = False
	while(i>=0):
		index = order.index(prev)
		for x in range((index+1), (index+9)):
			#print order[x]
			if(abs(coords[0]-current) <2 and abs(coords[1]-maxcoordy)<2 and approx < 0 ):
				exit = True
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
			     	contour.append([coords[0]+order[x%8][0],coords[1]+order[x%8][1]])
			     	#print coords
			     	#if(coords in visited1):
                             	#    visited2.append(coords)
                             	#else:
                             	#   visited1.append(coords)
			     	#visited1.append(coords)
			     
			     	break
                             	#print point
		
		i = i-1
		approx = approx-1
		if(exit == True):
			break
	#print len(contour)
	ctr = np.array(contour).reshape((-1,1,2)).astype(np.int32)
	cv2.drawContours(img,[ctr],0,(255,255,255),1)
	#print coords
	#below finds center spot via checking of dark sq


	#cv2.imshow("imageedit", newImg)
	
	
        try:
	        (x,y),(MA,ma),angle = cv2.fitEllipse(ctr)
        except:
                return False
        origangle=angle
	angle = angle/180*3.14
	#x = int(x)
	#y = int(y)
	#cv2.ellipse(img, (x,y),(MA,ma),angle)
	#print angle
	tupleEnd = (int(round(x+MA*math.cos(angle)/2)),int(round(y-MA*math.sin(angle)/2)))
	tupleStart = (int(round(x-MA*math.cos(angle)/2)),int(round(y+MA*math.sin(angle)/2)))
	m = round(math.sin(angle)/math.cos(angle))
	b = int(round(y-m*x))
	m= int(m)
	xcenter =x
	ycenter = y
	cv2.line(img, tupleStart,tupleEnd,(255,255,255))
	#cv2.line(newImg, tupleStart,tupleEnd,(255,255,255))
	#cv2.imshow("imageedit", newImg)
	#image at this point has contour and major axis
	def retxy(x,y,m,b):
		if(y<x*m+b):
			return 1 
		if(y>x*m+b):
			return 2
		if(y==x*m+b):
			return 3

	asym = np.zeros((len(img),len(img[0]),3), np.uint8)
	asym[:,:,:] = (255,255,255)    
	cv2.drawContours(asym, ctr, -1, (0,0,0), 2)

	mask = np.zeros((len(img)+2,len(img[0])+2,3), np.uint8)
	mask[:,:,:] = (255,255,255)
	cv2.drawContours(asym, ctr, -1, (0,0,0), 2)
	mask = np.zeros(mask.shape[:2], np.uint8)

	cv2.floodFill(asym,mask,(maxcoordx,maxcoordy),(0,0,0))
	perim= cv2.arcLength(ctr,True)
	area = cv2.contourArea(ctr)
	irreg=float(perim)*perim/4/math.pi/area

	#M = cv2.getRotationMatrix2D((len(img[0]),len(img)),origangle,1)
	#dst = cv2.warpAffine(asym,M,(len(img[0]),len(img)))
	dst = imutils.rotate_bound(asym, origangle)

	maskRotate = np.zeros((len(dst)+2,len(dst[0])+2,3), np.uint8)
	maskRotate[:,:,:] = (255,255,255)
	cv2.drawContours(maskRotate, ctr, -1, (0,0,0), 2)
	maskRotate = np.zeros(maskRotate.shape[:2], np.uint8)

	cv2.floodFill(dst,maskRotate,(0,0),(255,255,255))
	cv2.floodFill(dst,maskRotate,(0,len(dst)-1),(255,255,255))
	cv2.floodFill(dst,maskRotate,(len(dst[0])-1,0),(255,255,255))
	cv2.floodFill(dst,maskRotate,(len(dst[0])-1,len(dst)-1),(255,255,255))

	for y in range(len(dst)):
        	for x in range(len(dst[0])):
			remove = True
			if(dst[y][x][0] != 255):
				for j in range(5):
					for i in range(5):
						if(y+j-2>=0 and y+j-2<len(dst) and x+i-2 >=0 and x+i-2<len(dst[0]) and dst[y+j-2][x+i-2][0]==0):
							remove = False
			if(remove == True):
				dst[y][x][0] = 255
				dst[y][x][1] = 255
				dst[y][x][2] = 255

	graydst = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(graydst,127,255,cv2.THRESH_BINARY)
	graydst, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	max = 0
	index=0
	for x in range(len(contours)):
		if(cv2.contourArea(contours[x])>max and cv2.contourArea(contours[x])<len(graydst[0])*len(graydst[0])*0.9):
			max = (cv2.contourArea(contours[x]))
			index = x
	#print index
	#print contours
	#cv2.fitEllipse(contours[0]) 
	#max1 = 
	#for i in range(len(contours)):
	
	(xDst,yDst),(MADst,maDst),angleDst = cv2.fitEllipse(contours[index])
	graydst = cv2.cvtColor(graydst,cv2.COLOR_GRAY2BGR)

	tupleEndDst = (int(round(xDst+MADst*math.cos(angleDst)/2)),int(round(yDst-MADst*math.sin(angleDst)/2)))
	tupleStartDst = (int(round(xDst-MADst*math.cos(angleDst)/2)),int(round(yDst+MADst*math.sin(angleDst)/2)))
	#leftmost = tuple(contours[index][contours[index][:,:,0].argmin()][0])
	#rightmost = tuple(contours[index][contours[index][:,:,0].argmax()][0])
	topmost = tuple(contours[index][contours[index][:,:,1].argmin()][0])
	bottommost = tuple(contours[index][contours[index][:,:,1].argmax()][0])
	yavg = (bottommost[1]+topmost[1])/2
	cv2.line(graydst, (0,yavg),(len(graydst[0]),yavg),(255,255,255))
	#print cv2.contourArea(contours[0])
	#cv2.drawContours(graydst, contours, 2, (0,255,0), 3)
	currentArea = cv2.contourArea(contours[index])

	top = np.zeros((len(dst),len(dst[0]),3), np.uint8)
	top[:,:,:] = (255,255,255)

	bottom = np.zeros((len(dst),len(dst[0]),3), np.uint8)
	bottom[:,:,:] = (255,255,255)

	comb = np.zeros((len(dst),len(dst[0]),3), np.uint8)
	comb[:,:,:] = (255,255,255)

	#print asym[0][0]
	for y in range(len(graydst)):
		for x in range(len(graydst[0])):
			if(y<yavg):
				bottom[y][x] = graydst[y][x]
			if(y>yavg):
				top[y][x] = graydst[y][x]
			if(y==yavg):
				bottom[y][x] = graydst[y][x]
				top[y][x] = graydst[y][x]
	bottom =cv2.flip(bottom,0)
	bottom = cv2.cvtColor(bottom,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(bottom,127,255,cv2.THRESH_BINARY_INV)
	bottom, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	bottom = cv2.cvtColor(bottom,cv2.COLOR_GRAY2BGR)
	max = 0
	index=0
	for x in range(len(contours)):
        	if(cv2.contourArea(contours[x])>max and cv2.contourArea(contours[x])<len(graydst[0])*len(graydst[0])*0.9):
                	max = (cv2.contourArea(contours[x]))
                	index = x
	topmost = tuple(contours[index][contours[index][:,:,1].argmin()][0])
	x = index
	while(bottom[topmost[1]][x][0]>0):
		x = x-1
	topLeftBot = [x,topmost[1]]

	#M = np.float32([[1,0,-topLeftBot[0]+200],[0,1,-topLeftBot[1]]])
	#bottom = cv2.warpAffine(bottom,M,(len(bottom[0]),len(bottom)))

	top = cv2.cvtColor(top,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(top,127,255,cv2.THRESH_BINARY_INV)
	top, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	top = cv2.cvtColor(top,cv2.COLOR_GRAY2BGR)
	max = 0
	index=0
	for x in range(len(contours)):
        	if(cv2.contourArea(contours[x])>max and cv2.contourArea(contours[x])<len(graydst[0])*len(graydst[0])*0.9):
                	max = (cv2.contourArea(contours[x]))
                	index = x
	topmost = tuple(contours[index][contours[index][:,:,1].argmin()][0])
	x = index
	while(top[topmost[1]][x][0]>0):
        	x = x-1
	topLeftTop = [x,topmost[1]]



	M = np.float32([[1,0,-100],[0,1,-topLeftBot[1]+5]])
	bottom = cv2.warpAffine(bottom,M,(len(bottom[0]),len(bottom)))

	M = np.float32([[1,0,-100],[0,1,-topLeftTop[1]+5]])
	top = cv2.warpAffine(top,M,(len(bottom[0]),len(bottom)))

	comb = copy.deepcopy(top)
	for y in range(len(comb)):
        	for x in range(len(comb[0])):
			if(bottom[y][x][0] != 0 and comb[y][x][0] != 0):
				comb[y][x][1] = 0
			if(bottom[y][x][0] != 0 and comb[y][x][0] == 0):
				comb[y][x][0] =255
			if(bottom[y][x][0] == 0 and comb[y][x][0] != 0):
				comb[y][x][1] = 0
				comb[y][x][0] = 0
	finalArea=copy.deepcopy(comb)
	for y in range(len(finalArea)):
        	for x in range(len(finalArea[0])):
			if(finalArea[y][x][0] == 0 and finalArea[y][x][2] == 255 ):
				finalArea[y][x][2]=0
			
			if(finalArea[y][x][0] == 255 and finalArea[y][x][2] == 0 ):
				finalArea[y][x][0]=0
			if(finalArea[y][x][0] == 255 and finalArea[y][x][2] == 255 ):
                        	finalArea[y][x][1]=255

	finalArea = cv2.cvtColor(finalArea,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(finalArea,127,255,cv2.THRESH_BINARY)
	finalArea, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	finalArea = cv2.cvtColor(finalArea,cv2.COLOR_GRAY2BGR)
	max = 0
	index=0
	for x in range(len(contours)):
        	if(cv2.contourArea(contours[x])>max and cv2.contourArea(contours[x])<len(graydst[0])*len(graydst[0])*0.9):
                	max = (cv2.contourArea(contours[x]))
                	index = x
	x = index

	symArea = cv2.contourArea(contours[index])
	final=float(symArea)/currentArea*100
	#cv2.imwrite(os.path.join(os.path.expanduser('~'),"ScienceFair2018","examples","symmetry",filename+"sym.jpg"),finalArea)
	#print("Border Irregularity: ")
	filename = filename.split("/")
	path = ""
	for x in range(len(filename)-1):
		path =  path +filename[x]+"/"
	now = datetime.datetime.now()
	time = now.strftime("%Y-%m-%d at %H.%M.%S")
	cv2.line(asym, tupleStart,tupleEnd,(255,255,255))
	cv2.imwrite(path+time+"-border.png", newImg)    
	cv2.imwrite(path+time+"-overlap.png",comb)
	#cv2.imwrite(path+"cont.png",graydst)
        
	cv2.imwrite(path+time+"-majoraxis.png",asym)
	cv2.imwrite(path+time+"-finalArea.png",finalArea)

	return [irreg,final]
	#print("symmetry: ")
#run("Images/ISIC_0000016.png",1.0)

#cv2.line(asym, tupleStart,tupleEnd,(255,255,255))


#cv2.imshow("comb",comb)
#cv2.imshow("top",top)
#cv2.imshow("bot",bottom)
#cv2.imshow("ctr",asym)
#cv2.imshow("cont", graydst)
#cv2.imshow("final", finalArea)

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
#cv2.waitKey(0)
#cv2.destroyAllWindows()


