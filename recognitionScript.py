import cv2
import numpy as np
import string
import trainingHelper

###############   training part    ############### 

#Input: 
#	Sample file,Response File
#Output: 
#	KNearest neighboor Model made from learning from the two files
def train(samplesFile, responsesFile):
	samples = np.loadtxt(samplesFile,np.float32)
	responses = np.loadtxt(responsesFile,np.float32)
	responses = responses.reshape((responses.size,1))

	model = cv2.KNearest()
	model.train(samples,responses)
	return model

############################# testing part  #########################

numlist =  map(str, range(10))
alphalistl = list(string.ascii_lowercase)
alphalistC = list(string.ascii_lowercase)
answers = alphalistC+alphalistl+numlist  

#Input:
# 	Image file
# 	Machine Learning Training Model
def manRec(im, model, answers = []):
	ansLen = (len(answers) > 0)
	ind = 0 # Answer index used to match rect to answer
	correct = 1.0 # start at 1 to avoid divide by 0 error
	notc = 0
	size = 10

	out = np.zeros(im.shape,np.uint8)
	
	# Convert image to Grayscale	
	gray = trainingHelper.color2gray(im)
	
	# Gaussian blur the Image to help edge detection
	blur= trainingHelper.gaussianBlur(gray)

	# Run an adaptive threshold on Grayscale Image
	thresh = trainingHelper.adapThreshold(blur)
	
	# Find the Countours of White letters in the Black background of thresh (a grayscale image)
	contours,hierarchy = trainingHelper.findCountours(thresh)

	# Convert outline of Countours into  4 point rectangles
	rectangles = trainingHelper.findCountourAreas(contours,5) # leave out 1 if not combining tibble
	
	# Remove the rectangles that overlap with each other since no 2 letters overlap on each other
	rectangles = trainingHelper.removeOverlaps(rectangles)

	# Sort rectangles so you can read them left to right, top to bottom
	rectangles = trainingHelper.xsort(rectangles)

	# Merge tibble of the I and J
	# rectangles = trainingHelper.mergeT(rectangles) # combine tibble 

	#For each rectangles identify the most likely character it resembles 
	for (x,y,w,h) in rectangles:

		# Show a light  green Rectangle over onces we are about to process/have
		cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
		# Waitkey() before a show is required to update it
		cv2.waitKey(33)
		cv2.imshow('im',im)

		roi = thresh[y:y+h,x:x+w]
		#Number of extra values to append to size vector
		num_Extra_Vals=0

		# Resize Region of Intrest into a size*size matrix of intesity values
		# The default is 10 by 10 matrix
		roismall = cv2.resize(roi,(size,size))

		# Resize Pixel Intensity Matrix into a vector
		# default is a 100 value vector of intensities
		roismall = roismall.reshape((1,size*size + num_Extra_Vals))

		# p25 = sum(roismall[0][0:size*size/2])*1.0/(size*size/2)
		# p26 = sum(roismall[0][size*size/2+1:size*size])*1.0/(size*size/2)
		# p27 = (w*255.0/h )
		# roismall[0].append([p25,p25,p27])
		# roismall=[np.append(roismall[0],[p25,p25,p27],0)]

		# Convert vector into a float
		roismall = np.float32(roismall)

		# Find nearest vector using the Model's nearest KNN algorithm
		retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
		
		# Character that the ROI was recognized as
		string = str(chr((results[0][0])))
		
		#if there are answers to compare we display  them
		if (ansLen):
			if(string==answers[ind%len(answers)]):
				cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
				correct+=1
			else:

				cv2.putText(out,string,(x,y+h),0,1,(0,0,255))
				notc+=1
				
		else:
			cv2.putText(out,string,(x,y+h),0,1,(0,255,255))

		ind+=1

	print "Accuracy: ", correct/(correct+notc)


	# Show results and wait until a key is pressed to exit
	cv2.imshow('im',im)
	cv2.imshow('out',out)
	cv2.waitKey(0)

#Input:
#	Filename
#Output:
#	Runs the demo
def run(fileName,answersList=[]):
	im = cv2.imread(fileName)
	model = train('generalsamples.data', 'generalresponses.data')
	print manRec(im,model,answersList)

run('MachineLearning/alphabet.png',answers)

