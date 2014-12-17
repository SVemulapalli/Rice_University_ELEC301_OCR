from os import listdir, getcwd
from os.path import isfile, join, splitext
import sys

import cv2
import numpy as np
import bisect 

#Input:
# path to directory
#Output
#	Sorted list of all files in a given directory
def getAllFiles(directory=getcwd()):
	onlyfiles = sorted([ f for f in listdir(directory) if isfile(join(directory,f)) ]) # only add files in directory, don't include folders
	return onlyfiles

#Input: 
#	path directory for a Image file 
#Output: 
#	Tuple containing the image object and a copy
def readImage(fileName):
	im = cv2.imread(fileName)
	imcopy = im.copy()
	print "Read file:", fileName
	return im, imcopy
#Input: 
#	path directory for a Image file 
#Output: 
#	Tuple containing the image object and a copy
def readFilename(filename):
	justName = splitext(filename)[0] # remove extension
	justName = justName.lower() # convert to lowercase
	asciiKey = ord(justName)
	return justName, asciiKey

#Input: 
#	Image object that is to be converted
#Output: 
#	Converts the color image to grayscale
def color2gray(image):
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # convert color image to grayscale
	return gray

#Input:
# 	Image object that is to be detected
#Output:
# 	Draw lines and return list of regions where there are similar colors and sizes
def mser(image):
	regions = cv2.MSER().detect(gray, None)
	hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
	cv2.polylines(image, hulls, 1, (0, 255, 0))
	return regions

#Input: 
#	Image object that is to be converted
#Output: 
#	Image objected blurred to the specified parameters
def gaussianBlur(image, width=5, height=5, xStdev=0, yStdev=0): # width and height has to be odd and positive
	gblur = cv2.GaussianBlur(image,(width,height),xStdev) # gaussian blur
	return gblur

# Input: 
#	 Image object that has been grayscaled
# (Optional) Parameters:
# 	MaxValue -  max pixel value, 255 in this case
# 	Adaptive Method - It decides how thresholding value is calculated.
#			cv2.ADAPTIVE_THRESH_MEAN_C : threshold value is the mean of neighbourhood area.
#			cv2.ADAPTIVE_THRESH_GAUSSIAN_C : threshold value is the weighted sum of neighbourhood values where weights are a gaussian window.
# 	Theshold value- style of threshold, in this case Binary thresh
#			cv2.THRESH_BINARY
#			cv2.THRESH_BINARY_INV
#			cv2.THRESH_TRUNC
#			cv2.THRESH_TOZERO
#			cv2.THRESH_TOZERO_INV
# Block Size - decides the size of neighbourhood area.
# C - constant which is subtracted from the mean or weighted mean calculated.
#
# Output: 
#	same type of image object but where each pixel has gone through binary
# Further Reading:http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
def adapThreshold(image, MaxValue=255, Method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, Threshold_Type = cv2.THRESH_BINARY_INV, Blocksize=11,c=2):

	thresh = cv2.adaptiveThreshold(image, MaxValue, Method, Threshold_Type, Blocksize, c)
	return thresh

# Input: 
#	 Image object that will be modified to find contours,Will find white objects in this black file
#	(Optional) Parameters:
#	Approx Method	
#		cv2.CHAIN_APPROX_SIMPLE: returns a simple amount of points i.e 4 for a box for the corners
#		cv2.CHAIN_APPROX_NONE: Returns all reduntant points for countour  i.e hundres for a box for points on edge
#	Retrival Method 		
#		cv2.CV_RETR_EXTERNAL retrieves only the extreme outer contours. It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours.
# 		cv2.CV_RETR_LIST retrieves all of the contours without establishing any hierarchical relationships.
# 		cv2.CV_RETR_CCOMP retrieves all of the contours and organizes them into a two-level hierarchy. At the top level, there are external boundaries of the components. At the second level, there are boundaries of the holes. If there is another contour inside a hole of a connected component, it is still put at the top level.
# 		cv2.CV_RETR_TRE
# Output:
# 	Countours: list of all the contours in the image
#	Heirarchy: output vector, containing information about the image topology and countours. It has as many elements as the number of contours.
# Further Reading:
#	http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html?highlight=contours#what-are-contours
#	http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#cv2.findContours
def findCountours(image,Contour_Rtrval_Md =cv2.RETR_LIST,Contour_Approx_Method = cv2.CHAIN_APPROX_SIMPLE):
	contours,hierarchy = cv2.findContours(image,Contour_Rtrval_Md,Contour_Approx_Method)
	return contours, hierarchy

# Input: 
#	 countours:list of  countours of a Image
# 	(optional) Parameters:
#		h: optional minimun height of rectangles
# Output:
# 	Countours: list of all the rechtangle contours where
#		the height of the rectangles is greater than minH
# Further Reading:
#	http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html?highlight=contours#what-are-contours
def findCountourAreas(contours, minH=5):
	rectangles= []

	for cnt in contours:
		if cv2.contourArea(cnt)>10:
			#Calculates the up-right bounding rectangle of a point set.
			[x,y,w,h] = cv2.boundingRect(cnt)

			if h>minH:
				rectangles.append((x,y,w,h))
	return rectangles

# Input: 
#	 Rect1,Rect2: (x,y) bound of a rectangle
# Output:
# 	Boolean indicating if Rectangles overlap
def intersect(rect1, rect2):
	# right
	aright=rect1[0]+rect1[2]
	bright=rect2[0]+rect2[2]
	# top
	atop= rect1[1]+rect1[3]	
	btop=rect2[1]+rect2[3]
	# bottom
	abottom=rect1[1]
	bbottom=rect2[1]
	# left
	aleft=rect1[0]
	bleft=rect2[0]

	separate = aright < bleft or aleft > bright or atop < bbottom or abottom > btop

	return not separate

# Input: 
#	 Rectangles: List of rectangles contours
# Output:
# 	A copy of the Input where overlapping rectangles have been removed
def removeOverlaps(rectangles):
	badindices=[]

	for i in (range(0,len(rectangles))):
		for j in (range(i,len(rectangles))):
			
			rect1 = rectangles[i]
			rect2 = rectangles[j]
			if (i!=j):
				if(intersect(rect1,rect2)):
					smallerrect=i;
					if(rect2[2]*rect2[3]<rect1[2]*rect1[3]):
						smallerrect= j	
					if(not (smallerrect in badindices)):
						bisect.insort(badindices,i)

	for x in reversed(badindices):
		rectangles.pop(x)

	return rectangles

# Input: 
#	 Rectangles: List of rectangles contours
# 	 (Optional) Tibble_gap: Possible pixel vertical gap between only i and j's tibbles,
# Output:
# 	A copy of the Input where rectangles with Tibble_gap vertically away are merged
def mergeT(orginal,Tibble_gap=5):
	rectangles = []
	merged = []
	newlist=[]
	for n in orginal:
		rectangles.append(n)

	for i in xrange(0,len(rectangles)):
		
		for j in xrange(0,len(rectangles)):	
			if(i not in merged and j not in merged and i!=j):
				(x,y,w,h)=rectangles[i]
				(y2,x2,w2,h2)=rectangles[j]
				rect1=rectangles[j]
				rect2=(x,y,w,h+Tibble_gap)
				tibble=j
				nontibble=i
				if(True==intersect(rect1,rect2)):
					# cv2.rectangle(im,(x,y-5),(x+w,y+h+5),(0,133,133),1)

					if(w*h<w2*h2):
						nontibble=j
						tibble=i
					gap =rectangles[nontibble][1]-( rectangles[tibble][1]+ rectangles[tibble][3]  )
					nh=rectangles[nontibble][3]+(rectangles[tibble][3] + gap )
					ny=rectangles[nontibble][1]-(rectangles[tibble][3] + gap )
					nrect=(rectangles[nontibble][0],ny,rectangles[nontibble][2],nh)
					rectangles[nontibble]=nrect

					bisect.insort(merged,tibble)

	for k in reversed(merged):
		rectangles.pop(k)
	return rectangles

#Input:
#	Rectangles: List of rectangles contours
# 	Simple sorting by either lowest x then y value or vice-versa, [0:1] = [x:y]
#Output:
#	Sorted list of rectangles
def sortListedRect(rectangles,a=1,b=0):
	rectangles = sorted(rectangles, key=lambda x: (x[a],x[b])) # sort by x,y values
	return rectangles